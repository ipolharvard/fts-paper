import inspect
import math
from enum import StrEnum

import torch


class ModelType(StrEnum):
    DECODER = "decoder"
    ENC_DECODER = "encoder-decoder"


def make_infinite_loader(loader):
    while True:
        yield from iter(loader)


def get_lr(it, args):
    """Learning rate decay scheduler (cosine with warmup)."""
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.lr * it / args.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return args.min_lr + coeff * (args.lr - args.min_lr)


def configure_optimizers(model: torch.nn.Module, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    sum(p.numel() for p in decay_params)
    sum(p.numel() for p in nodecay_params)
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and "cuda" in device_type
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    return optimizer


def estimate_mfu(model, num_params, fwdbwd_per_iter, dt):
    """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
    # first estimate the number of flops we do per iteration.
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    cfg = model.config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.n_positions
    flops_per_token = 6 * num_params + 12 * L * H * Q * T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0 / dt)  # per second
    flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu


def get_num_params(self, non_embedding=True):
    """Return the number of parameters in the model.

    For non-embedding count (default), the position embeddings get subtracted. The token embeddings
    would too, except due to the parameter sharing these params are actually used as weights in the
    final layer, so we include them.
    """
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
        n_params -= self.wpe.weight.numel()
    return n_params
