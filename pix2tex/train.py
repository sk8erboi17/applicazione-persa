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
from torch.utils.data import WeightedRandomSampler
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

    def _fill_queue(self):
        try:
            for seq, im in self.dataloader:
                if self._stop:
                    break
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


def create_scheduler(opt, args):
    """Create learning rate scheduler with warmup support."""
    scheduler_name = args.get('scheduler', 'StepLR')

    if scheduler_name == 'CosineAnnealingWarmRestarts':
        T_0 = args.get('T_0', 5000)
        T_mult = args.get('T_mult', 2)
        eta_min = args.get('eta_min', 1e-6)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

    if scheduler_name == 'CosineAnnealingLR':
        T_max = args.get('T_max', args.epochs * 1000)
        eta_min = args.get('eta_min', 1e-6)
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)

    # Default: StepLR
    return get_scheduler(scheduler_name)(opt, step_size=args.get('lr_step', 30), gamma=args.get('gamma', 0.9995))


class WarmupScheduler:
    """Wraps a scheduler to add linear warmup."""

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

    if args.load_chkpt is not None:
        model.load_state_dict(torch.load(args.load_chkpt, map_location=device))

    def save_models(e, step=0):
        ckpt_path = os.path.join(out_path, '%s_e%02d_step%02d.pth' % (args.name, e+1, step))
        torch.save(model.state_dict(), ckpt_path)
        yaml.dump(dict(args), open(os.path.join(out_path, 'config.yaml'), 'w+'))
        print(f"\n  Checkpoint saved: {ckpt_path}")

    # Enhanced optimizer and scheduler
    opt = create_optimizer(model, args)
    base_scheduler = create_scheduler(opt, args)

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

    global_step = 0

    # Print training config summary
    print("\n" + "=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"  Device:       {device}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batchsize} (micro: {microbatch})")
    print(f"  LR:           {args.lr}")
    print(f"  Optimizer:    {args.get('optimizer', 'Adam')}")
    print(f"  Scheduler:    {args.get('scheduler', 'StepLR')}")
    print(f"  Warmup:       {warmup_steps} steps")
    print(f"  AMP:          {use_amp}")
    print(f"  Grad clip:    {gradient_clip}")
    print(f"  Early stop:   {early_stopping_patience} evals")
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
                    if score > best_score:
                        best_score = score
                        max_bleu, max_token_acc = bleu_score, token_accuracy
                        save_models(e, step=i)
                        no_improvement_count = 0
                        print(f"  >>> New best: BLEU={max_bleu:.4f}, ACC={max_token_acc:.4f}")
                    else:
                        no_improvement_count += 1

                    # Early stopping
                    if early_stopping_patience > 0 and no_improvement_count >= early_stopping_patience:
                        print(f"\nEarly stopping after {no_improvement_count} evaluations without improvement")
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
