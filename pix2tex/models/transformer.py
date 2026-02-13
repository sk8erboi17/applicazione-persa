import torch
import torch.nn.functional as F
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from x_transformers import TransformerWrapper, Decoder

# x-transformers v2.x moved/renamed top_k and top_p
try:
    from x_transformers.autoregressive_wrapper import top_k, top_p
except ImportError:
    top_k = top_p = None


def _top_k(logits, k=0.9):
    """Top-k filtering compatible with both old and new x-transformers."""
    if isinstance(k, float):
        # Fraction of vocab — convert to int
        num_tokens = logits.shape[-1]
        k = max(int(k * num_tokens), 1)
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits


class CustomARWrapper(AutoregressiveWrapper):
    def __init__(self, *args, label_smoothing=0.0, **kwargs):
        super(CustomARWrapper, self).__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing

    def forward(self, x, **kwargs):
        """Forward pass with label smoothing support.

        Overrides AutoregressiveWrapper.forward() to pass label_smoothing
        to F.cross_entropy (the base class ignores the config value).
        """
        xi = x[:, :-1]
        xo = x[:, 1:]
        # Adjust mask to match shifted sequence length
        mask = kwargs.get('mask', None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs['mask'] = mask
        out = self.net(xi, **kwargs)
        loss = F.cross_entropy(
            out.transpose(1, 2), xo,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )
        return loss

    @torch.no_grad()
    def generate(self, start_tokens, seq_len=256, eos_token=None, temperature=1., filter_thres=0.9, **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop('mask', None)
        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]
            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]

            if temperature == 0:
                # Greedy decoding: deterministic, best for eval
                sample = logits.argmax(dim=-1, keepdim=True)
            else:
                # Stochastic sampling with top-k filtering
                if filter_logits_fn in {top_k, top_p}:
                    filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                    probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out


def get_decoder(args):
    return CustomARWrapper(
        TransformerWrapper(
            num_tokens=args.num_tokens,
            max_seq_len=args.max_seq_len,
            attn_layers=Decoder(
                dim=args.dim,
                depth=args.num_layers,
                heads=args.heads,
                **args.decoder_args
            )),
        pad_value=args.pad_token,
        label_smoothing=args.get('label_smoothing', 0.0))
