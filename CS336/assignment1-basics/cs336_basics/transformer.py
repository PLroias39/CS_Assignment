# transformer.py
# task_02
#

import torch
import torch.nn as nn
from einops import rearrange, einsum

class Linear(nn.Module):
    """
    input: x=[5.001, ..., 5.512]: in_features=512
    output: x=[5.0001, ..., 5.1024]: out_features=1024
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        in_features: int .final dimension of the input
        out_features: int .final dimension of the output
        device: torch.device | None = None .Device to store the parameters on
        dtype: torch.dtype | None = None .Data type of the parameters
        '''
        # subclass nn.Module
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        # parameters
        self.weight = nn.Parameter(torch.empty(
            out_features,
            in_features,
            device = device,
            dtype = dtype
        ))
        self._init_weight()

    def _init_weight(self):
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = x * W^T
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')

class Embedding(nn.Module):
    """
    map integer token_ids to a d_model dimension vector
    input: [[5, 12, 7], [45, 3, 39]]: batch_size=2, max_seq_len=3
    output: [[[5.01, 5.02, ..., 5.16], ...], ...]: batch_size=2, max_seq_len=3, embedding_dim=16
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        num_embeddings: int .Size of the vocabulary
        embedding_dim: int .Dimension of the embedding vectors, i.e.,
        device: torch.device | None = None .Device to store the parameters on
        dtype: torch.dtype | None = None .Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        # parameters
        self.weight = nn.Parameter(torch.empty(
            num_embeddings,
            embedding_dim,
            device = device,
            dtype = dtype
        ))
        self._init_weight()

    def _init_weight(self):
        torch.nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        input: token_ids -> [batch_size, max_seq_len]
        output: vector -> [batch_size, max_seq_len, embedding_dim]
        """
        token_ids = token_ids.to(torch.long) 
        return self.weight[token_ids]

def _test_embedding():
    emb = Embedding(num_embeddings=10, embedding_dim=16)
    # batch_size=2, max_seq_len=3
    # [25565, 3939, 45] as one sentence
    input = torch.tensor([[25565, 3939, 0], [4, 3, 2]])
    # token_embeds[0] = [
    # [0.10, 0.20, 0.30, ..., 1.60],    # word 25565's vector
    # [0.01, 0.02, 0.03, ..., 0.16],    # word 3939's vector
    # [0, ...]  # 0's vector
    # ]
    # each id expressed by 16 numbers
    output = emb(input)
    print(f"{output.shape}\n[INFO] output of sentence1: {output[0]}")

class RMSNorm(nn.Module):
    """
    input: (batch_size, max_seq_len, d_model)
    output: (batch_size, max_seq_len, d_model)
    """
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        '''
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        # weight
        self.weight = nn.Parameter(torch.empty(
            d_model,
            device = device,
            dtype = dtype
        ))
        self._init_weight()

    def _init_weight(self):
        torch.nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        # compute Root Mean Square
        ms = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        # devided by x_fp32
        x_normed = x_fp32 / rms
        # y = x^T * W 
        output = einsum(x_normed, self.weight, "... d_model, d_model -> ... d_model")
        
        return output.to(input_dtype)

class SwiGLU(nn.Module):
    """
    input: (batch_size, max_seq_len, d_model=512)
    hidden: (batch_size, max_seq_len, d_ff=2048)
    output: (batch_size, max_seq_len, d_model=512)
    """
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if d_ff == None:
            d_ff = int(8 * d_model) / 3
            d_ff = (d_ff + 63) // 64*64
        self.device  = device
        self.dtype = dtype
        self.w_gate = nn.Parameter(torch.empty(
            self.d_ff,
            self.d_model,
            device = device,
            dtype = dtype
        ))
        self.w_up = nn.Parameter(torch.empty(
            self.d_ff,
            self.d_model,
            device = device,
            dtype = dtype
        ))
        self.w_down = nn.Parameter(torch.empty(
            self.d_model,
            self.d_ff,
            device = device,
            dtype = dtype
        ))
        self._init_weight()

    def _init_weight(self):
        torch.nn.init.trunc_normal_(self.w_gate, std=0.02)
        torch.nn.init.trunc_normal_(self.w_up, std=0.02)
        torch.nn.init.trunc_normal_(self.w_down, std=0.02)
    
    def forward(self, x) -> torch.Tensor:
        # compute SiLU
        gate = einsum(self.w_gate, x, 'd_ff d_model, ... d_model -> ... d_ff')
        gate_activated = gate * torch.sigmoid(gate)
        # up cast
        up_proj = einsum(self.w_up, x, 'd_ff d_model, ... d_model -> ... d_ff')
        # position-wise multiply
        intermediate = gate_activated * up_proj
        
        # down cast
        return einsum(self.w_down, intermediate, 'd_model d_ff, ... d_ff -> ... d_model')

# designed for Attention computation
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        '''
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        if d_k % 2 != 0:
            raise ValueError("RoPE needs d_k must be even.")
        
        # prepare thetas and cos/sin list, and cache them
        indices = torch.arange(0, d_k, 2, device=device).float()
        thetas = theta ** (-indices / d_k)
        
        positions = torch.arange(max_seq_len, device=device).float()
        angles = einsum(positions, thetas, 'm, j -> m j')   # max_seq_len, d_k/2
        self.register_buffer("cos", angles.cos())   # (max_seq_len, d_k/2)
        self.register_buffer("sin", angles.sin())   # (max_seq_len, d_k/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        """
        *batch_size, seq_len, _ = x.shape

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        curr_cos = self.cos[token_positions]
        curr_sin = self.sin[token_positions]
        
        x_rearrange = rearrange(x, '... seq (d_half two) -> ... seq d_half two', two=2)
        x_1 = x_rearrange[..., 0]   # (..., max_seq_len, d_k/2)
        x_2 = x_rearrange[..., 1]
        out_1 = x_1 * curr_cos - x_2 * curr_sin
        out_2 = x_1 * curr_sin + x_2 * curr_cos

        out = torch.stack([out_1, out_2], dim=-1)

        return rearrange(out, "... seq d_half two -> ... seq (d_half two)").type_as(x)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x)
    return x / torch.sum(x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    query: (batch_size, ..., seq_len_q, d_k)
    key: (batch_size, ..., seq_len_k, d_k)
    value: (batch_size, ..., seq_len_k, d_v)
    """
    d_k = query.size(-1)
    scores = einsum(query, key, '... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k')
    scores = scores / d_k ** 0.5

    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))

    attn_weight = softmax(scores, dim=-1)
    output = einsum(attn_weight, value, '... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v')

    return output

# MHA
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope, max_seq_len:int, device=None, dtype=None):
        '''
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        use_causal_mask: bool Whether to apply causal masking.
        '''
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divided by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope
        self.max_seq_len = max_seq_len
        
        factory_kwargs = {"device": device, "dtype": dtype}

        """
        # optimization_1
        #
        self.w_query = Linear(d_model, d_model, **factory_kwargs)
        self.w_key = Linear(d_model, d_model, **factory_kwargs)
        self.w_value = Linear(d_model, d_model, **factory_kwargs)
        """
        self.w_qkv = Linear(d_model, 3*d_model, **factory_kwargs)
        self.w_out = Linear(d_model, d_model, **factory_kwargs)
        # causal_mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        """
        x: (*batch_size, max_seq_len, d_model)
        """
        *batch_size, seq_len, _ = x.shape
        """
        # optimization_1 
        #
        # get Linear layer of x
        q = self.w_query(x)
        k = self.w_key(x)
        v = self.w_value(x)
        # split to multihead
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)
        """
        # get qkv at once
        qkv = self.w_qkv(x) # shape: (..., seq_len, 3*d_model)
        qkv = rearrange(qkv, '... s (n h d) -> n ... h s d', n=3, h=self.num_heads)
        q, k ,v = qkv[0], qkv[1], qkv[2]
        # apply RoPE
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        # get mask
        mask = self.causal_mask[:seq_len, :seq_len]
        # compute Attention
        attn = scaled_dot_product_attention(q, k, v, mask=mask)
        output = rearrange(attn, '... h s d -> ... s (h d)')
        
        return self.w_out(output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope, max_seq_len:int, device=None, dtype=None):
        """
        d_model: int  Dimensionality of the Transformer block inputs.
        num_heads: int  Number of heads to use in multi-head self-attention.
        d_ff: int  Dimensionality of the position-wise feed-forward inner layer.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_ff = d_ff
        self.rope = rope
        # layer
        factory_kwargs = {"device" : device, "dtype" : dtype}
        self.ln1 = RMSNorm(
            d_model, 
            **factory_kwargs
        )
        self.attn = MultiheadSelfAttention(
            d_model = d_model, 
            num_heads = num_heads, 
            rope = rope, 
            max_seq_len = max_seq_len,
            **factory_kwargs
        )
        self.ln2 = RMSNorm(
            d_model, 
            **factory_kwargs
        )
        self.ffn = SwiGLU(
            d_model = d_model, 
            d_ff = d_ff, 
            **factory_kwargs
        )

    def forward(self, x: torch.Tensor, token_positions:torch.Tensor=None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        
        return x

class TransformerLM(nn.Module):
    """
    input: (*batch_size, seq_len), sequence of token_ids
    embed: (*batch_size, seq_len, embedding_dim),use hidden_dim to express each token_ids
    transformer: keep same dim
    norm: keep same dim
    Linea: (*batch_size, seq_len, vocab_size)
    """
    def __init__(self, vocab_size: int,context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: int,device=None, dtype=None):
        '''
        vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
        context_length: int The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
        num_layers: int The number of Transformer blocks to use.
        '''
        super().__init__() 
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        
        self.head_dim = d_model // num_heads
        factory_kwargs = {"device" : device, "dtype" : dtype}

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim = d_model,
            **factory_kwargs
        )
        self.rope = RotaryPositionalEmbedding(
            theta = rope_theta,
            d_k = self.head_dim,
            max_seq_len = context_length,
            device = device
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model = d_model,
                num_heads = num_heads,
                d_ff = d_ff,
                rope = self.rope,
                max_seq_len = context_length, 
                **factory_kwargs
            )for _ in range(num_layers)
        ])
        self.output_norm = RMSNorm(
            d_model = d_model,
            **factory_kwargs
        )
        self.output_embedding = Linear(
            in_features = d_model,
            out_features = vocab_size,
            **factory_kwargs
        )

    def forward(self, x: torch.Tensor):
        """
        x: (*batch_size, seq_len, d_model)
        """
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, token_positions=None)
        x = self.output_norm(x)
        x = self.output_embedding(x)
    
        return x

if __name__ == "__main__":
    pass
    # _test_embedding()

