# generate.py 
# task-06
#
# Input:
# prompt, model, tokenizer
# Output:
# result text
import torch
import argparse
import pathlib
import sys
from typing import List, Optional
from cs336_basics.transformer import TransformerLM, softmax
from cs336_basics.tokenization import Tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="CS336 Generation Script")
    # 模型配置 (需与训练时一致)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_len', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--rope_theta', type=float, default=10000.0)
    
    # 生成配置
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default="Once upon a time,", help='Prompt for generation')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p threshold')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 路径配置
    parser.add_argument('--vocab_path', type=str, default=None)
    parser.add_argument('--merge_path', type=str, default=None)
    
    return parser.parse_args()

def apply_temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        # 如果温度极低，采样逻辑通常会直接取 argmax，这里返回原始 logits 即可
        return logits
    return logits / temperature

def apply_top_p_filtering(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    对概率分布进行核采样（Top-p）过滤。
    
    参数:
        probs (torch.Tensor): 形状为 (vocab_size,) 的归一化概率分布。
        p (float): 累积概率阈值 (0 < p <= 1)。
    
    返回:
        torch.Tensor: 过滤后的概率分布，不在 Top-p 集合中的项概率被设为 0，并重新归一化。
    """
    if p >= 1.0:
        return probs
    
    # 1. 对概率进行降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 2. 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 3. 找到满足累积概率 >= p 的最小集合
    # 我们需要包含第一个使累积概率超过 p 的 token，所以将掩码向右移一位
    sorted_indices_to_remove = cumulative_probs > p
    # 确保第一个 token 始终不被移除（即使它的概率已经 > p）
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # 4. 将被移除的 token 概率设为 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    # [重要修复] 将 indices_to_remove 从 1D 变形为 2D 以匹配 probs 的维度
    indices_to_remove = indices_to_remove.unsqueeze(0)
    probs.scatter_(1, indices_to_remove, 0)
    
    # 5. 重新归一化
    return probs / probs.sum()

def sample_next_token(
    logits: torch.Tensor, 
    temperature: float = 1.0, 
    top_p: float = 1.0
) -> int:
    """
    根据 logits、温度和 Top-p 阈值采样下一个 token。
    
    参数:
        logits (torch.Tensor): 形状为 (vocab_size,) 的模型输出（最后一维）。
        temperature (float): 温度缩放参数。
        top_p (float): 核采样阈值。
        
    返回:
        int: 采样的 token ID。
    """
    # 1. 温度缩放
    # 如果温度非常近于 0，直接执行 Greedy Search (取最大值)
    if temperature < 1e-6:
        return torch.argmax(logits).item()
    
    scaled_logits = apply_temperature_scaling(logits, temperature)
    
    # 2. 转化为概率分布 (Softmax)
    probs = softmax(scaled_logits, dim=-1)
    
    # 3. Top-p 过滤
    if top_p < 1.0:
        probs = apply_top_p_filtering(probs, top_p)
    
    # 4. 采样
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()

@troch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    device: str = "cpu"
) -> List[int]:
    """
    
    """
    model.eval()
    generated = list(prompt_ids)
    context_length = model.context_length
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 1. 准备输入：取最近的 context_length 个 token
            input_ids = generated[-context_length:]
            input_tensor = torch.tensor([input_ids], device=device) # (1, seq_len)
            
            # 2. 模型前向传播
            # 输出形状: (1, seq_len, vocab_size)
            logits = model(input_tensor)
            
            # 3. 取最后一个位置的 logits 进行采样
            next_token_logits = logits[0, -1, :]
            
            # 4. 采样下一个 token
            next_token = sample_next_token(next_token_logits, temperature, top_p)
            
            # 5. 追加结果
            generated.append(next_token)
            
            # 6. 检查是否达到结束符
            if eos_token_id is not None and next_token == eos_token_id:
                break
                
    return generated


def main():
    args = parse_args()
    
    # 1. 加载分词器
    root_dir = pathlib.Path(__file__).resolve().parent.parent
    vocab_path = args.vocab_path or str(root_dir / "data" / "Tinystory_vocab.json")
    merge_path = args.merge_path or str(root_dir / "data" / "Tinystory_merges.txt")
    
    print(f"Loading tokenizer from {vocab_path}...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merge_path,
        special_tokens=["<|endoftext|>"]
    )
    
    # 2. 初始化模型
    print(f"Initializing model on {args.device}...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device
    )
    
    # 3. 加载权重
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    # 4. 准备 Prompt
    prompt_ids = tokenizer.encode(args.prompt)
    eos_token_id = tokenizer.byte_to_int.get(b"<|endoftext|>")
    
    print(f"\nPrompt: {args.prompt}")
    print(f"Generating (temp={args.temperature}, top_p={args.top_p})...\n")
    
    # 5. 执行生成
    output_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        device=args.device
    )
    
    # 6. 解码并打印
    output_text = tokenizer.decode(output_ids)
    print(f"Result:\n{output_text}")

if __name__ == "__main__":
    main()
