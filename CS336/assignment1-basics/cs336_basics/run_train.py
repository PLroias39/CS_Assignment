# run_train.py 
# task_05
# 
# Input:
# train.bin, valid.bin: data file
# TransformerLM: model file
# some function: to build training loop 

import torch
import numpy as np 
import pathlib

from cs336_basics.tokenization import Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.train import (cross_entropy_loss, AdamW, get_lr_cosine_schedule, 
    gradient_clipping, get_batch, save_checkpoint, load_checkpoint
)

import wandb
import argparse
import time
import random
from tqdm import tqdm
from typing import NamedTuple

# vocab and merges file from tokenization/run_train_bpe_v1
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

TRAIN_BIN = DATA_DIR / "TinyStoriesV2-GPT4-train.bin"
VALID_BIN = DATA_DIR / "TinyStoriesV2-GPT4-valid.bin"
TINY_BIN = DATA_DIR / "tiny_TinyStoriesV2-GPT4-train.bin"

def parse_args():
    parser = argparse.ArgumentParser(description="CS336 Assignment 1 Training")
    # Model Arguments, see class transformer/TransformerLM
    parser.add_argument('--vocab_size', type=int, default=10000, help='Size of vocabulary')
    parser.add_argument('--context_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1344, help='FFN dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto/cpu/cuda/mps')
    # Train Arguments, see file train.py 
    # -get_lr_cosine_schedule
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning rate')
    parser.add_argument('--warm_up_it', type=int, default=500, help='Warmup iterations')
    parser.add_argument('--cosine_it', type=int, default=10000, help='Cosine annealing iterations')
    # -AdamW 
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    # -gradient_clipping
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    # Training Loop Arguments, see this file
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--train_steps', type=int, default=6000, help='Total training steps')
    parser.add_argument('--val_interval', type=int, default=100, help='Validation interval')
    parser.add_argument('--val_batches', type=int, default=10, help='Number of validation batches')
    # -checkpoint
    parser.add_argument('--save_intervals', type=int, default=1000, help='Checkpoint save interval')
    parser.add_argument('--save_ckp_path', type=str, default='./checkpoints', help='Checkpoint save directory')
    parser.add_argument('--resume_ckp', type=str, default=None, help='Path to checkpoint to resume from')
    # All Data needed to use 
    parser.add_argument('--vocab_path', type=str, default=None, help='vocab data for tokenizer')
    parser.add_argument('--merge_path', type=str, default=None, help='merge data for tokenizer')
    parser.add_argument('--train_data', type=str, default=None, help='data for train')
    parser.add_argument('--valid_data', type=str, default=None, help='data for valid')
    """Wandb"""
    parser.add_argument('--log_intervals', type=int, default=1, help='Logging interval')
    parser.add_argument('--wandb_log', action="store_true", help="Enable logging to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="cs336_assignment1")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()

def set_environment(args, seed=3939):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"\n[Init] Device set to: {device}")
    print(f"[Init] Seed set to: {seed}")
    return device


""" Part 2: Create dataset memmap """
def get_dataset_memmap(bin_path: pathlib.Path, dtype=np.uint16) -> np.memmap:
    """
    map Binary data to memory
    """
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary data not found: {bin_path}")

    print(f"\n[Init] Dataset loaded from: {bin_path}")
    return np.memmap(bin_path, dtype=dtype, mode='r')


""" Part 3: Handle checkpoint loading """
def handle_load_checkpoint(resume_ckp_path: str | pathlib.Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    start_iter = 0 
    if resume_ckp_path is None or resume_ckp_path == "": 
        return start_iter
    if not resume_ckp_path.exists():
        print(f"[\nWarning] Checkpoint not found at: {resume_ckp_path}, start from scratch")
        return start_iter

    print(f"[\nInfo] Resume checkpoint from: {resume_ckp_path}")
    start_iter = load_checkpoint(resume_ckp_path, model, optimizer) # please ensure on cpu 
    print(f"[Done] checkpoint resume at: {start_iter}")
    return start_iter


@torch.no_grad()
def evaluate(model, data, args, device):
    model.eval()
    print(f"\n[Info] Start evaluation mode")
    losses = []
    for _ in range(args.val_batches):
        x, y = get_batch(data, args.batch_size, args.context_len, device)
        logits = model(x)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    avg_loss = np.mean(losses)
    ppl = np.exp(avg_loss)
    print(f"[Done] Evaluation done")
    model.train()
    return avg_loss, ppl


def main():
    args = parse_args()
    device = set_environment(args, seed=3939)
    if args.wandb_log:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"d{args.d_model}-l{args.num_layers}-h{args.num_heads}",
            config=vars(args)
        )

    # Note 1: model & optimizer & tokenizer
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device
    ).to(device)
    
    optimizer = AdamW(
        params=model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
     
    # Note 2: dataset
    train_data = pathlib.Path(args.train_data) if args.train_data else TINY_BIN
    valid_data = pathlib.Path(args.valid_data) if args.valid_data else VALID_BIN
    train_data = get_dataset_memmap(train_data)
    valid_data = get_dataset_memmap(valid_data)

    # Note 3: checkpoint loading
    start_iter = handle_load_checkpoint(args.resume_ckp, model, optimizer)
    
    # Note 4: start training
    model.train()
    train_losses = []
    pbar = tqdm(range(start_iter, args.train_steps),desc="Training")
    for it in pbar:
        # Get learning rate
        lr = get_lr_cosine_schedule(
            it=it, 
            max_learning_rate=args.max_lr, 
            min_learning_rate=args.min_lr, 
            warmup_iters=args.warm_up_it, 
            cosine_cycle_iters=args.cosine_it
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # Get batch data
        X, Y = get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_len,
            device=device
        )
        # Feedward & Backward boardcast
        optimizer.zero_grad()
        logits = model(X)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), Y.view(-1))
        loss.backward()

        # Gradient clip 
        if args.clip_grad_norm > 0:
            gradient_clipping(model.parameters(), args.clip_grad_norm)
        optimizer.step()

        # Logging
        if it % args.log_intervals == 0:
            train_metrics = {
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/step": it 
            }
            if args.wandb_log:
                wandb.log(train_metrics)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "lr": f"{lr:.2e}"
            })

        # Evaluation
        if it > 0 and it % args.val_interval == 0:
            val_loss, val_ppl = evaluate(model, valid_data, args, device)
            print(f"[Eval] Step {it}: Val loss {val_loss:.4f}, PPL {val_ppl:.2f}")
            val_metrics = {
                "val/loss": val_loss,
                "val/ppl": val_ppl,
                "train/step": it 
            }
            if args.wandb_log:
                wandb.log(val_metrics)

        # Save checkpoint
        if it > 0 and it % args.save_intervals == 0:
            ckpt_path = pathlib.Path(args.save_ckp_path) / f"ckpt_step_{it}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, it, ckpt_path)
    
    if args.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    main()
    

