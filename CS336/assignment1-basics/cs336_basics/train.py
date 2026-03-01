# train.py
# task_03

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Iterable, Optional, Union, BinaryIO, IO 
import math 
import numpy as np 
import os 

def cross_entropy_loss(out_logit: torch.Tensor, target: torch.Tensor):
    '''
    out_logit: (batch_size, vocab_size), out_logit[i][j] means unnormalized logit of jth class for the ith example.
    target: (batch_size), target[i] means the index of correct class
    '''
    # torch.logsumexp()
    max_logits, _ = torch.max(out_logit, dim=1, keepdim=True)
    log_sum_exp = torch.log(torch.sum(torch.exp(out_logit - max_logits), dim=1, keepdim=True)) + max_logits
    # log_probs
    log_probs = out_logit - log_sum_exp
    # extract correct logit
    target_log_probs = log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
    loss = -target_log_probs.mean()

    return loss


class AdamW(Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float=1e-3, betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-8, weight_decay: float=0.01):
        if lr < 0.0:    raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:   raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:   raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps <= 0.0:  raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:  raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        # loop1, for each group in groups 
        for group in self.param_groups:
            # loop2, for each param in this group 
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                self._update_param(p, group)
        
        return loss

    def _update_param(self, p, group):
        """
        
        """
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0 
            state["exp_avg"] = torch.zeros_like(p)      # m_t 
            state["exp_avg_sq"] = torch.zeros_like(p)   # v_t
        
        # get hyperparams
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        # get state
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        state["step"] += 1 
        t = state["step"]
        
        grad = p.grad
        # ----- computation -----
        # weight_decay
        p.mul_(1 - lr * weight_decay)
        # update m, v 
        exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

        bias_correction1 = 1 - beta1 ** t 
        bias_correction2 = 1 - beta2 ** t 

        step_size = lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
        # update p 
        p.addcdiv_(exp_avg, denom, value=-step_size)


def get_lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    """
    linear warmup + cosine decay 
    [0, warmup_iters]: linear increase to max_lr 
    [warmup_iters, cosine_cycle_iters]: cosine decrease to min_lr 
    [cosine_cycle_iters, inf]: keep min_lr
    """ 
    # stage 1: warmup
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    # stage 3: post-decay
    if it >= cosine_cycle_iters:
        return min_learning_rate
    # stage 2: cosine decay
    decay_radio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosff = 0.5 * (1.0 + math.cos(math.pi * decay_radio))
    lr = min_learning_rate + cosff * (max_learning_rate - min_learning_rate)
    
    return lr


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # filter params that have not grad 
    params = [p for p in parameters if p.grad is not None]
    if not params: 
        return
    # compute norm for all params
    total_norm_sq = 0.0 
    for p in params:
        # use p.grad.detach() to don't compute graph
        param_norm_sq = torch.sum(p.grad.detach() ** 2)
        total_norm_sq += param_norm_sq
    total_norm = torch.sqrt(total_norm_sq)
    # compute scale factor
    eps = 1e-6  # avoid devided by zero 
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for p in params:
            p.grad.detach().mul_(clip_coef)


def get_batch(dataset: np.typing.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    dataset: np.memmap
    dataset = np.memmap('data/train.bin', dtype=np.uint16, mode='r')
    """
    n = len(dataset)
    max_start_idx = n - context_length - 1
    # random sample some start points 
    start_ids = np.random.randint(0, max_start_idx + 1, size=batch_size)
    # generate data 
    x_list = [dataset[i: i + context_length] for i in start_ids]
    y_list = [dataset[i+1: i + context_length + 1] for i in start_ids]
    # transfer to Tensor & move to device
    x = torch.from_numpy(np.array(x_list)).to(dtype=torch.long, device=device)
    y = torch.from_numpy(np.array(y_list)).to(dtype=torch.long, device=device)

    return x, y 


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: Union[str, os.PathLike, BinaryIO, IO[bytes]]):
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint_data, out)

def load_checkpoint(src: Union[str, os.PathLike, BinaryIO, IO[bytes]], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']

    return int(iteration)


