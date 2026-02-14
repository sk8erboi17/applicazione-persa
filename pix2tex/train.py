from pix2tex.dataset.dataset import Im2LatexDataset
import os
import json
import argparse
import logging
import yaml
import math
from threading import Thread
from queue import Queue

import torch
from munch import Munch
from tqdm.auto import tqdm
import wandb
import torch.nn as nn
from pix2tex.eval import evaluate
from pix2tex.models import get_model
from pix2tex.utils import in_model_path, parse_args, seed_everything, get_optimizer, get_scheduler, gpu_memory_check


class PrefetchIterator:
    """Prefetch next batch in a background thread while GPU trains on current batch."""

    def __init__(self, dataloader, device, prefetch_size=2):
        self.dataloader = dataloader
        self.device = device
        self.prefetch_size = prefetch_size
        self.queue = Queue(maxsize=prefetch_size)
        self.thread = None
        self._stop = False
        self._use_pin_memory = 'cuda' in str(device)

    def _fill_queue(self):
        try:
            for seq, im in self.dataloader:
                if self._stop:
                    break
                # Pin memory for faster H2D transfer (safe in background thread)
                if self._use_pin_memory and im is not None:
                    im = im.pin_memory()
                self.queue.put((seq, im))
            self.queue.put(None)  # Sentinel
        except Exception:
            self.queue.put(None)

    def __iter__(self):
        self._stop = False
        self.thread = Thread(target=self._fill_queue, daemon=True)
        self.thread.start()
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

    def __len__(self):
        return len(self.dataloader)


def get_device_auto(args):
    """Detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available() and not args.get('no_cuda', False):
        return args.device  # Already set by parse_args
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def create_optimizer(model, args):
    """Create optimizer with support for AdamW and weight decay."""
    optimizer_name = args.get('optimizer', 'Adam')
    lr = args.lr
    betas = tuple(args.betas)
    weight_decay = args.get('weight_decay', 0.0)

    if optimizer_name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    return get_optimizer(optimizer_name)(model.parameters(), lr, betas=betas)


def create_scheduler(opt, args, steps_per_epoch=None):
    """Create learning rate scheduler."""
    scheduler_name = args.get('scheduler', 'StepLR')

    if scheduler_name == 'OneCycleLR':
        total_steps = (steps_per_epoch or 8000) * args.epochs
        return torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr,
            total_steps=total_steps,
            pct_start=0.05,            # 5% warmup
            anneal_strategy='cos',
            div_factor=25,             # initial_lr = max_lr / 25
            final_div_factor=1000      # final_lr = initial_lr / 1000
        )

    if scheduler_name == 'CosineAnnealingWarmRestarts':
        T_0 = args.get('T_0', 5000)
        T_mult = args.get('T_mult', 2)
        eta_min = args.get('eta_min', 1e-6)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

    if scheduler_name == 'CosineAnnealingLR':
        total_steps = (steps_per_epoch or 8000) * args.epochs
        warmup = args.get('warmup_steps', 0)
        T_max = args.get('T_max', total_steps - warmup)
        eta_min = args.get('eta_min', 1e-6)
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)

    # Default: StepLR
    return get_scheduler(scheduler_name)(opt, step_size=args.get('lr_step', 30), gamma=args.get('gamma', 0.9995))


class WarmupScheduler:
    """Wraps a scheduler to add linear warmup. Not needed with OneCycleLR."""

    def __init__(self, optimizer, scheduler, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            warmup_factor = self.current_step / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.base_lr * warmup_factor
        else:
            self.scheduler.step()

    def get_last_lr(self):
        if self.current_step <= self.warmup_steps:
            return [self.base_lr * self.current_step / self.warmup_steps]
        return self.scheduler.get_last_lr() if hasattr(self.scheduler, 'get_last_lr') else [self.base_lr]

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'base_lr': self.base_lr,
            'scheduler': self.scheduler.state_dict(),
        }

    def load_state_dict(self, state):
        self.current_step = state['current_step']
        self.warmup_steps = state['warmup_steps']
        self.base_lr = state['base_lr']
        self.scheduler.load_state_dict(state['scheduler'])


def load_sample_weights(weights_path):
    """Load sample weights from JSON for weighted random sampling.

    Returns dict with 'weights' list and metadata, or None if not found.
    """
    if weights_path and os.path.exists(weights_path):
        with open(weights_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded sample weights: {data.get('n_printed', '?')} printed (w={data.get('w_printed', 1):.2f}), "
              f"{data.get('n_handwritten', '?')} handwritten (w={data.get('w_handwritten', 1):.2f}), "
              f"target HW ratio: {data.get('target_hw_ratio', '?')}")
        return data
    return None


def train(args):
    dataloader = Im2LatexDataset().load(args.data)
    dataloader.update(**args, test=False)
    valdataloader = Im2LatexDataset().load(args.valdata)
    valargs = args.copy()
    valargs.update(batchsize=args.testbatchsize, keep_smaller_batches=True, test=True)
    valdataloader.update(**valargs)

    # Device selection with MPS support
    device = get_device_auto(args)
    args.device = device
    print(f"Using device: {device}")

    # Load weighted sampling config (for printed/handwritten balancing)
    weights_data = load_sample_weights(args.get('sample_weights', None))
    if weights_data and 'weights' in weights_data:
        dataloader.sample_weights = weights_data['weights']
        print(f"  Weighted sampling enabled: {len(weights_data['weights'])} weights loaded")

    model = get_model(args)
    if torch.cuda.is_available() and not args.get('no_cuda', False):
        gpu_memory_check(model, args)

    # CUDA optimizations
    if 'cuda' in str(device):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')  # Enable TF32 on tensor cores (~20% speedup)
        torch.cuda.empty_cache()

    # torch.compile for MPS/CUDA acceleration (PyTorch 2.0+)
    if hasattr(torch, 'compile') and device in ('mps', 'cuda'):
        try:
            compile_backend = 'inductor' if 'cuda' in str(device) else None
            model = torch.compile(model, backend=compile_backend) if compile_backend else torch.compile(model)
            print(f"Model compiled with torch.compile(backend={compile_backend})")
        except Exception as e:
            print(f"torch.compile not supported: {e}")

    max_bleu, max_token_acc = 0, 0
    best_score = 0
    out_path = os.path.join(args.model_path, args.name)
    os.makedirs(out_path, exist_ok=True)

    # Load model weights (supports both old format and new full checkpoint)
    _resume_ckpt = None
    if args.load_chkpt is not None:
        _resume_ckpt = torch.load(args.load_chkpt, map_location=device)
        if isinstance(_resume_ckpt, dict) and 'model' in _resume_ckpt:
            model.load_state_dict(_resume_ckpt['model'])
            print(f"  Loaded model from {args.load_chkpt} (full checkpoint)")
        else:
            model.load_state_dict(_resume_ckpt)
            _resume_ckpt = None  # Old format, nothing else to restore
            print(f"  Loaded model from {args.load_chkpt} (weights only)")

    def save_models(e, step=0):
        ckpt_path = os.path.join(out_path, '%s_e%02d_step%02d.pth' % (args.name, e+1, step))
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': e,
            'global_step': global_step,
            'best_score': best_score,
            'no_improvement_count': no_improvement_count,
            'eval_count': eval_count,
        }
        torch.save(checkpoint, ckpt_path)
        yaml.dump(dict(args), open(os.path.join(out_path, 'config.yaml'), 'w+'))
        print(f"\n  Checkpoint saved: {ckpt_path}")

    # Compute steps per epoch for scheduler
    iter(dataloader)  # Trigger __iter__ to compute size
    steps_per_epoch = len(dataloader)
    print(f"  Steps/epoch:  {steps_per_epoch}")

    # Enhanced optimizer and scheduler
    opt = create_optimizer(model, args)
    scheduler_name = args.get('scheduler', 'StepLR')

    if scheduler_name == 'OneCycleLR':
        # OneCycleLR has built-in warmup, no WarmupScheduler needed
        scheduler = create_scheduler(opt, args, steps_per_epoch=steps_per_epoch)
    else:
        base_scheduler = create_scheduler(opt, args, steps_per_epoch=steps_per_epoch)
        warmup_steps = args.get('warmup_steps', 0)
        if warmup_steps > 0:
            scheduler = WarmupScheduler(opt, base_scheduler, warmup_steps, args.lr)
        else:
            scheduler = base_scheduler

    microbatch = args.get('micro_batchsize', -1)
    if microbatch == -1:
        microbatch = args.batchsize

    # Mixed precision (AMP) — use bfloat16 on CUDA (more stable on Ampere+), float16 on MPS
    use_amp = args.get('use_amp', False)
    amp_dtype = torch.bfloat16 if 'cuda' in str(device) else torch.float16
    # Determine autocast device type
    if 'cuda' in str(device):
        autocast_device = 'cuda'
    elif device == 'mps':
        autocast_device = 'mps'
    else:
        autocast_device = 'cpu'
    scaler = torch.amp.GradScaler(enabled=(use_amp and autocast_device == 'cuda'))

    # Gradient clipping
    gradient_clip = args.get('gradient_clip', 1.0)

    # Early stopping
    early_stopping_patience = args.get('early_stopping_patience', 0)
    no_improvement_count = 0
    # Guards: warmup grace period + loss-trend veto
    early_stopping_warmup_evals = args.get('early_stopping_warmup_evals', 0)
    early_stopping_loss_patience = args.get('early_stopping_loss_patience', 0)
    eval_count = 0
    recent_losses = []
    prev_window_avg_loss = None
    loss_is_decreasing = False

    global_step = 0

    # Restore full training state from checkpoint (if available)
    if _resume_ckpt is not None and 'optimizer' in _resume_ckpt:
        opt.load_state_dict(_resume_ckpt['optimizer'])
        scheduler.load_state_dict(_resume_ckpt['scheduler'])
        scaler.load_state_dict(_resume_ckpt['scaler'])
        args.epoch = _resume_ckpt['epoch'] + 1
        global_step = _resume_ckpt.get('global_step', 0)
        best_score = _resume_ckpt.get('best_score', 0)
        no_improvement_count = _resume_ckpt.get('no_improvement_count', 0)
        eval_count = _resume_ckpt.get('eval_count', 0)
        print(f"  Restored training state: epoch={args.epoch}, step={global_step}, "
              f"best={best_score:.4f}, patience={no_improvement_count}/{early_stopping_patience}")
        del _resume_ckpt  # Free memory

    # Print training config summary
    print("\n" + "=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"  Device:       {device}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batchsize} (micro: {microbatch})")
    print(f"  LR:           {args.lr}")
    print(f"  Optimizer:    {args.get('optimizer', 'Adam')}")
    print(f"  Scheduler:    {scheduler_name}")
    warmup_steps_cfg = args.get('warmup_steps', 0)
    if warmup_steps_cfg > 0:
        print(f"  Warmup:       {warmup_steps_cfg} steps")
    if scheduler_name == 'CosineAnnealingLR':
        total_s = steps_per_epoch * args.epochs
        t_max = args.get('T_max', total_s - warmup_steps_cfg)
        print(f"  T_max:        {t_max} (total: {total_s})")
        print(f"  eta_min:      {args.get('eta_min', 1e-6)}")
    print(f"  AMP:          {use_amp}")
    print(f"  Grad clip:    {gradient_clip}")
    print(f"  Early stop:   {early_stopping_patience} evals (warmup: {early_stopping_warmup_evals}, loss_patience: {early_stopping_loss_patience})")
    print(f"  Steps/epoch:  {steps_per_epoch}")
    if weights_data:
        print(f"  HW sampling:  target {weights_data.get('target_hw_ratio', 0.2):.0%}")
    print("=" * 50 + "\n")

    try:
        for e in range(args.epoch, args.epochs):
            args.epoch = e
            prefetch = PrefetchIterator(iter(dataloader), device, prefetch_size=5)
            dset = tqdm(prefetch, desc=f"Epoch {e+1}/{args.epochs}", total=len(dataloader))
            for i, (seq, im) in enumerate(dset):
                if seq is not None and im is not None:
                    opt.zero_grad()
                    total_loss = 0

                    for j in range(0, len(im), microbatch):
                        tgt_seq = seq['input_ids'][j:j+microbatch].to(device, non_blocking=True)
                        tgt_mask = seq['attention_mask'][j:j+microbatch].bool().to(device, non_blocking=True)
                        images = im[j:j+microbatch].to(device, non_blocking=True)

                        if use_amp and autocast_device in ('cuda', 'mps'):
                            with torch.amp.autocast(device_type=autocast_device, dtype=amp_dtype):
                                loss = model.data_parallel(
                                    images, device_ids=args.gpu_devices,
                                    tgt_seq=tgt_seq, mask=tgt_mask
                                ) * microbatch / args.batchsize
                            if autocast_device == 'cuda':
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()
                        else:
                            loss = model.data_parallel(
                                images, device_ids=args.gpu_devices,
                                tgt_seq=tgt_seq, mask=tgt_mask
                            ) * microbatch / args.batchsize
                            loss.backward()

                        total_loss += loss.item()

                    # Gradient clipping
                    if use_amp and autocast_device == 'cuda':
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                    # Optimizer step
                    if use_amp and autocast_device == 'cuda':
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()

                    scheduler.step()
                    global_step += 1

                    # Track loss for early stopping loss-trend veto
                    if early_stopping_loss_patience > 0:
                        recent_losses.append(total_loss)

                    # Logging
                    current_lr = opt.param_groups[0]['lr']
                    dset.set_description(
                        f'Epoch {e+1}/{args.epochs} | Loss: {total_loss:.4f} | LR: {current_lr:.2e}'
                    )
                    if args.wandb:
                        wandb.log({
                            'train/loss': total_loss,
                            'train/lr': current_lr,
                            'train/global_step': global_step,
                        })

                if (i+1+len(dataloader)*e) % args.sample_freq == 0:
                    bleu_score, edit_distance, token_accuracy = evaluate(
                        model, valdataloader, args,
                        num_batches=args.valbatches,
                        name='val'
                    )
                    if 'cuda' in str(device):
                        torch.cuda.empty_cache()  # Free VRAM after validation
                    score = bleu_score + token_accuracy
                    eval_count += 1

                    # Compute loss trend for this eval window
                    if early_stopping_loss_patience > 0 and len(recent_losses) > 0:
                        curr_avg = sum(recent_losses) / len(recent_losses)
                        if prev_window_avg_loss is not None:
                            loss_is_decreasing = curr_avg < prev_window_avg_loss * 0.995
                        prev_window_avg_loss = curr_avg
                        recent_losses = []

                    if score > best_score:
                        best_score = score
                        max_bleu, max_token_acc = bleu_score, token_accuracy
                        save_models(e, step=i)
                        no_improvement_count = 0
                        print(f"  >>> New best: BLEU={max_bleu:.4f}, ACC={max_token_acc:.4f}")
                    else:
                        # Guard 1: warmup grace period
                        if eval_count <= early_stopping_warmup_evals:
                            print(f"  (warmup grace {eval_count}/{early_stopping_warmup_evals}, not counting)")
                        # Guard 2: loss still decreasing → model is learning
                        elif early_stopping_loss_patience > 0 and loss_is_decreasing:
                            print(f"  (loss still decreasing: {prev_window_avg_loss:.4f}, not counting)")
                        else:
                            no_improvement_count += 1

                    # Early stopping
                    if early_stopping_patience > 0 and no_improvement_count >= early_stopping_patience:
                        print(f"\nEarly stopping after {no_improvement_count} evals without improvement (eval #{eval_count})")
                        save_models(e, step=i)
                        return

            if (e+1) % args.save_freq == 0:
                save_models(e, step=len(dataloader))
            if args.wandb:
                wandb.log({'train/epoch': e+1})

    except KeyboardInterrupt:
        if e >= 2:
            save_models(e, step=i)
        raise KeyboardInterrupt
    save_models(e, step=len(dataloader))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', default=None, help='path to yaml config file', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')
    parser.add_argument('--debug', action='store_true', help='DEBUG')
    parser.add_argument('--resume', help='path to checkpoint folder', action='store_true')
    parsed_args = parser.parse_args()
    if parsed_args.config is None:
        with in_model_path():
            parsed_args.config = os.path.realpath('settings/debug.yaml')
    with open(parsed_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params), **vars(parsed_args))
    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    seed_everything(args.seed)
    if args.wandb:
        if not parsed_args.resume:
            args.id = wandb.util.generate_id()
        wandb.init(config=dict(args), resume='allow', name=args.name, id=args.id)
        args = Munch(wandb.config)
    train(args)
