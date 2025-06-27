from collections import defaultdict
from collections.abc import Callable

import numpy as np
import torch as th
from transformers import PreTrainedModel


@th.no_grad()
def estimate_loss(
    model: PreTrainedModel, ctx, get_batch: Callable, eval_iters: int, tokens_of_interest: dict
):
    block_size = 2048
    block_thresh = block_size // 2
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = th.empty(eval_iters)

        all_tokens_res = defaultdict(list)
        toi_res = defaultdict(list)

        for i in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                if isinstance(X, tuple):
                    output = model(input_ids=X[0], decoder_input_ids=X[1], labels=Y)
                else:
                    output = model(input_ids=X, labels=Y)
                loss, logits = output.loss, output.logits
            losses[i] = loss.item()

            if split == "val":
                for k in [1, 3, 5]:
                    all_tokens_res[f"acc_top/all/{split}/{block_thresh}/k={k}"].append(
                        top_k_accuracy(logits, Y, k=k, threshold=block_thresh)
                    )
                for stoken, token in tokens_of_interest.items():
                    for k in [1, 3, 10]:
                        toi_res[f"acc_top/{stoken}/{split}/{block_thresh}/k={k}"].append(
                            top_k_acc_special(logits, Y, token, k=k, threshold=block_thresh)
                        )
        out[f"loss/{split}"] = losses.mean()
        if split == "val":
            out.update({test_name: np.mean(v) for test_name, v in all_tokens_res.items()})
            out.update({test_name: compute_weighted_mean(v) for test_name, v in toi_res.items()})
    model.train()
    return out


def compute_weighted_mean(v):
    weights = sum(x[1] for x in v)
    if weights == 0:
        return th.nan
    return sum(x[0] * x[1] for x in v) / weights


@th.jit.script
def top_k_accuracy(logits: th.Tensor, y: th.Tensor, k: int, threshold: int) -> float:
    # logits: (batch_size, block_size, vocab_size)
    logits = logits[:, threshold - 1 :, :]
    y_true = y[:, threshold - 1 :, None]
    k = min(k, logits.size(-1))
    _, indices = th.topk(logits, k)
    correct = (indices == y_true).any(dim=-1)
    return float(correct.sum() / correct.numel())


@th.jit.script
def top_k_acc_special(
    logits: th.Tensor, y: th.Tensor, token_of_interest: int, k: int, threshold: int
) -> tuple[float, int]:
    logits = logits[:, threshold - 1 :, :]
    y_true = y[:, threshold - 1 :, None]
    interested = y_true == token_of_interest
    if interested.sum() == 0:
        return 0.0, 0
    _, indices = th.topk(logits, min(k, logits.size(-1)))
    correct = (indices == y_true).any(dim=-1, keepdim=True)
    weight = interested.sum()
    score = (correct & interested).sum() / weight
    return score.item(), weight.item()
