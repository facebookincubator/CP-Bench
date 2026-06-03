# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Op registry for PyTorch operator-level SDC detection.

To add a new op, decorate a factory function with @register_op:

    @register_op("MyOp", "my_category")
    def _my_op(detector, size):
        def op():
            a = detector.create_test_tensor(size)
            return torch.some_op(a)
        return op

To add a composite pipeline test, use @register_pipeline:

    @register_pipeline("My Pipeline")
    def _my_pipeline(detector, size):
        def pipeline():
            x = detector.create_test_tensor(size)
            x = torch.op1(x)
            x = torch.op2(x)
            return x
        return pipeline
"""

import os
from typing import Callable, List, NamedTuple

import torch


# ---------------------------------------------------------------------------
# Registry types
# ---------------------------------------------------------------------------


class OpDef(NamedTuple):
    name: str
    category: str
    factory: Callable  # (detector, size) -> callable


class PipelineDef(NamedTuple):
    name: str
    factory: Callable  # (detector, size) -> callable


OP_REGISTRY: List[OpDef] = []
PIPELINE_REGISTRY: List[PipelineDef] = []

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def register_op(name: str, category: str):
    """Decorator to register an op test.

    The decorated function must have signature (detector, size) -> callable,
    where the returned callable takes no args and returns a torch.Tensor.
    """

    def decorator(fn):
        OP_REGISTRY.append(OpDef(name=name, category=category, factory=fn))
        return fn

    return decorator


def register_pipeline(name: str):
    """Decorator to register a composite pipeline test."""

    def decorator(fn):
        PIPELINE_REGISTRY.append(PipelineDef(name=name, factory=fn))
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Arithmetic ops
# ---------------------------------------------------------------------------


@register_op("Addition", "arithmetic")
def _add_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        c = b.create_test_tensor(size)
        return a + c

    return op


@register_op("Multiplication", "arithmetic")
def _mul_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        c = b.create_test_tensor(size)
        return a * c

    return op


@register_op("Division", "arithmetic")
def _div_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        c = b.create_test_tensor(size) + 1.0
        return a / c

    return op


@register_op("Subtraction", "arithmetic")
def _sub_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        c = b.create_test_tensor(size)
        return a - c

    return op


@register_op("Power", "arithmetic")
def _pow_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return torch.pow(a, 2)

    return op


@register_op("Square Root", "arithmetic")
def _sqrt_op(b, size):
    def op():
        a = torch.abs(b.create_test_tensor(size))
        return torch.sqrt(a)

    return op


@register_op("Exponential", "arithmetic")
def _exp_op(b, size):
    def op():
        a = b.create_test_tensor(size) * 0.1
        return torch.exp(a)

    return op


@register_op("Logarithm", "arithmetic")
def _log_op(b, size):
    def op():
        a = torch.abs(b.create_test_tensor(size)) + 1.0
        return torch.log(a)

    return op


# ---------------------------------------------------------------------------
# Matrix ops
# ---------------------------------------------------------------------------


@register_op("Matrix Multiplication (matmul)", "matrix")
def _matmul_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        c = b.create_test_tensor((size[1], size[0]))
        return torch.matmul(a, c)

    return op


@register_op("Transpose", "matrix")
def _transpose_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return a.t()

    return op


@register_op("Batch Matrix Multiplication", "matrix")
def _bmm_op(b, size):
    def op():
        batch_size = 32
        m, n = size[0] // 4, size[1] // 4
        a = b.create_test_tensor((batch_size, m, n))
        c = b.create_test_tensor((batch_size, n, m))
        return torch.bmm(a, c)

    return op


@register_op("Matrix Multiplication (mm)", "matrix")
def _mm_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        c = b.create_test_tensor((size[1], size[0]))
        return torch.mm(a, c)

    return op


# ---------------------------------------------------------------------------
# Reduction ops
# ---------------------------------------------------------------------------


@register_op("Sum", "reduction")
def _sum_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return a.sum()

    return op


@register_op("Mean", "reduction")
def _mean_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return a.mean()

    return op


@register_op("Max", "reduction")
def _max_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return a.max()

    return op


@register_op("Min", "reduction")
def _min_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return a.min()

    return op


@register_op("Standard Deviation", "reduction")
def _std_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return a.std()

    return op


@register_op("Variance", "reduction")
def _var_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return a.var()

    return op


# ---------------------------------------------------------------------------
# Activation ops
# ---------------------------------------------------------------------------


@register_op("ReLU", "activation")
def _relu_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return torch.relu(a)

    return op


@register_op("Sigmoid", "activation")
def _sigmoid_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return torch.sigmoid(a)

    return op


@register_op("Tanh", "activation")
def _tanh_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return torch.tanh(a)

    return op


@register_op("Softmax", "activation")
def _softmax_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return torch.softmax(a, dim=-1)

    return op


@register_op("GELU", "activation")
def _gelu_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return torch.nn.functional.gelu(a)

    return op


@register_op("SiLU", "activation")
def _silu_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return torch.nn.functional.silu(a)

    return op


# ---------------------------------------------------------------------------
# Convolution ops
# ---------------------------------------------------------------------------


@register_op("Conv2D", "convolution")
def _conv2d_op(b, size):
    def op():
        x = b.create_test_tensor((16, 64, 32, 32))
        conv = torch.nn.Conv2d(64, 128, 3, padding=1).to(
            dtype=b.current_dtype, device=b.device
        )
        torch.manual_seed(42)
        conv.weight.data = torch.randn_like(conv.weight.data)
        conv.bias.data = torch.randn_like(conv.bias.data)
        return conv(x)

    return op


@register_op("Conv1D", "convolution")
def _conv1d_op(b, size):
    def op():
        x = b.create_test_tensor((32, 64, 128))
        conv = torch.nn.Conv1d(64, 128, 3, padding=1).to(
            dtype=b.current_dtype, device=b.device
        )
        torch.manual_seed(42)
        conv.weight.data = torch.randn_like(conv.weight.data)
        conv.bias.data = torch.randn_like(conv.bias.data)
        return conv(x)

    return op


# ---------------------------------------------------------------------------
# Memory ops
# ---------------------------------------------------------------------------


@register_op("Clone", "memory")
def _clone_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return a.clone()

    return op


@register_op("Copy", "memory")
def _copy_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        dst = torch.empty_like(a)
        dst.copy_(a)
        return dst

    return op


@register_op("Clone Large (14kx14k)", "memory")
def _clone_large_op(b, size):
    def op():
        torch.manual_seed(42)
        a = torch.randn(14000, 14000, dtype=b.current_dtype, device=b.device)
        return a.clone()

    return op


@register_op("Copy Large (14kx14k)", "memory")
def _copy_large_op(b, size):
    def op():
        torch.manual_seed(42)
        a = torch.randn(14000, 14000, dtype=b.current_dtype, device=b.device)
        dst = torch.empty_like(a)
        dst.copy_(a)
        return dst

    return op


@register_op("Reshape", "memory")
def _reshape_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return a.reshape(-1)

    return op


@register_op("View", "memory")
def _view_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return a.view(-1)

    return op


@register_op("Permute", "memory")
def _permute_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return a.permute(1, 0)

    return op


@register_op("Concatenate", "memory")
def _cat_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        c = b.create_test_tensor(size)
        return torch.cat([a, c], dim=0)

    return op


# ---------------------------------------------------------------------------
# Normalization ops
# ---------------------------------------------------------------------------


@register_op("LayerNorm", "normalization")
def _layernorm_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        ln = torch.nn.LayerNorm(size[1]).to(dtype=b.current_dtype, device=b.device)
        torch.manual_seed(42)
        ln.weight.data = torch.randn_like(ln.weight.data)
        ln.bias.data = torch.randn_like(ln.bias.data)
        return ln(a)

    return op


@register_op("BatchNorm2D", "normalization")
def _batchnorm_op(b, size):
    def op():
        x = b.create_test_tensor((32, 64, 32, 32))
        bn = torch.nn.BatchNorm2d(64).to(dtype=b.current_dtype, device=b.device)
        bn.eval()
        torch.manual_seed(42)
        bn.weight.data = torch.randn_like(bn.weight.data)
        bn.bias.data = torch.randn_like(bn.bias.data)
        return bn(x)

    return op


@register_op("GroupNorm", "normalization")
def _groupnorm_op(b, size):
    def op():
        x = b.create_test_tensor((16, 64, 32, 32))
        gn = torch.nn.GroupNorm(8, 64).to(dtype=b.current_dtype, device=b.device)
        torch.manual_seed(42)
        gn.weight.data = torch.randn_like(gn.weight.data)
        gn.bias.data = torch.randn_like(gn.bias.data)
        return gn(x)

    return op


@register_op("RMSNorm", "normalization")
def _rmsnorm_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        rms = torch.nn.RMSNorm(size[1]).to(dtype=b.current_dtype, device=b.device)
        torch.manual_seed(42)
        rms.weight.data = torch.randn_like(rms.weight.data)
        return rms(a)

    return op


# ---------------------------------------------------------------------------
# Advanced ops
# ---------------------------------------------------------------------------


@register_op("Dropout (eval mode)", "advanced")
def _dropout_op(b, size):
    def op():
        torch.manual_seed(42)
        a = b.create_test_tensor(size)
        return torch.nn.functional.dropout(a, p=0.5, training=False)

    return op


@register_op("Einsum", "advanced")
def _einsum_op(b, size):
    def op():
        a = b.create_test_tensor((128, 256))
        c = b.create_test_tensor((256, 128))
        return torch.einsum("ij,jk->ik", a, c)

    return op


# ---------------------------------------------------------------------------
# Attention ops
# ---------------------------------------------------------------------------


@register_op("Scaled Dot-Product Attention", "attention")
def _sdpa_op(b, size):
    def op():
        seq_len = min(size[0], 512)
        q = b.create_test_tensor((4, 8, seq_len, 64))
        k = b.create_test_tensor((4, 8, seq_len, 64))
        v = b.create_test_tensor((4, 8, seq_len, 64))
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    return op


# ---------------------------------------------------------------------------
# Scatter / Gather ops
# ---------------------------------------------------------------------------


@register_op("Scatter Add", "scatter_gather")
def _scatter_add_op(b, size):
    def op():
        torch.manual_seed(42)
        src = b.create_test_tensor(size)
        idx = torch.randint(0, size[0], (size[0], size[1]), device=b.device)
        out = torch.zeros(size, device=b.device, dtype=b.current_dtype)
        return out.scatter_add(0, idx, src)

    return op


@register_op("Gather", "scatter_gather")
def _gather_op(b, size):
    def op():
        torch.manual_seed(42)
        a = b.create_test_tensor(size)
        idx = torch.randint(0, size[0], (size[0], size[1]), device=b.device)
        return torch.gather(a, 0, idx)

    return op


@register_op("Index Select", "scatter_gather")
def _index_select_op(b, size):
    def op():
        torch.manual_seed(42)
        a = b.create_test_tensor(size)
        idx = torch.randint(0, size[0], (size[0] // 2,), device=b.device)
        return torch.index_select(a, 0, idx)

    return op


@register_op("Index Add", "scatter_gather")
def _index_add_op(b, size):
    def op():
        torch.manual_seed(42)
        a = b.create_test_tensor(size)
        idx = torch.arange(size[0] // 2, device=b.device)
        src = b.create_test_tensor((size[0] // 2, size[1]))
        return a.index_add(0, idx, src)

    return op


# ---------------------------------------------------------------------------
# Sorting ops
# ---------------------------------------------------------------------------


@register_op("Sort", "sorting")
def _sort_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return torch.sort(a, dim=-1).values

    return op


@register_op("TopK", "sorting")
def _topk_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return torch.topk(a, k=min(100, size[1]), dim=-1).values

    return op


# ---------------------------------------------------------------------------
# Embedding ops
# ---------------------------------------------------------------------------


@register_op("Embedding", "embedding")
def _embedding_op(b, size):
    def op():
        torch.manual_seed(42)
        emb = torch.nn.Embedding(10000, size[1]).to(b.device)
        idx = torch.randint(0, 10000, (32, 128), device=b.device)
        return emb(idx)

    return op


# ---------------------------------------------------------------------------
# Pooling ops
# ---------------------------------------------------------------------------


@register_op("MaxPool2D", "pooling")
def _maxpool2d_op(b, size):
    def op():
        x = b.create_test_tensor((16, 64, 32, 32))
        return torch.nn.MaxPool2d(kernel_size=2, stride=2)(x)

    return op


@register_op("AvgPool2D", "pooling")
def _avgpool2d_op(b, size):
    def op():
        x = b.create_test_tensor((16, 64, 32, 32))
        return torch.nn.AvgPool2d(kernel_size=2, stride=2)(x)

    return op


@register_op("AdaptiveAvgPool2D", "pooling")
def _adaptive_avgpool2d_op(b, size):
    def op():
        x = b.create_test_tensor((16, 64, 32, 32))
        return torch.nn.AdaptiveAvgPool2d((1, 1))(x)

    return op


# ---------------------------------------------------------------------------
# Indexing ops
# ---------------------------------------------------------------------------


@register_op("Masked Fill", "indexing")
def _masked_fill_op(b, size):
    def op():
        torch.manual_seed(42)
        a = b.create_test_tensor(size)
        mask = torch.randint(0, 2, size, device=b.device, dtype=torch.bool)
        # Use -1e4 instead of -1e9 to stay within float16 range (~65504)
        fill_val = -1e4 if b.current_dtype == torch.float16 else -1e9
        return a.masked_fill(mask, fill_val)

    return op


@register_op("Where", "indexing")
def _where_op(b, size):
    def op():
        torch.manual_seed(42)
        a = b.create_test_tensor(size)
        c = b.create_test_tensor(size)
        cond = torch.randint(0, 2, size, device=b.device, dtype=torch.bool)
        return torch.where(cond, a, c)

    return op


# ---------------------------------------------------------------------------
# Model-specific Linear (nn.Linear) ops
# ---------------------------------------------------------------------------
# These test the exact matmul shapes used by cpbench models.
# Models are dominated by nn.Linear which is matmul+bias -- the dominant
# compute op in every model. Non-square shapes exercise different GPU
# kernel tiling strategies than the existing square matmul tests.


@register_op("Linear BERT FFN Up (1024->4096)", "model_linear")
def _linear_bert_ffn_up(b, size):
    def op():
        torch.manual_seed(42)
        x = b.create_test_tensor((8192, 1024))
        linear = torch.nn.Linear(1024, 4096, bias=True).to(
            dtype=b.current_dtype, device=b.device
        )
        linear.weight.data = torch.randn_like(linear.weight.data)
        linear.bias.data = torch.randn_like(linear.bias.data)
        return linear(x)

    return op


@register_op("Linear BERT FFN Down (4096->1024)", "model_linear")
def _linear_bert_ffn_down(b, size):
    def op():
        torch.manual_seed(42)
        x = b.create_test_tensor((8192, 4096))
        linear = torch.nn.Linear(4096, 1024, bias=True).to(
            dtype=b.current_dtype, device=b.device
        )
        linear.weight.data = torch.randn_like(linear.weight.data)
        linear.bias.data = torch.randn_like(linear.bias.data)
        return linear(x)

    return op


@register_op("Linear GPT2 Fused QKV (1280->3840)", "model_linear")
def _linear_gpt2_qkv(b, size):
    def op():
        torch.manual_seed(42)
        x = b.create_test_tensor((8192, 1280))
        linear = torch.nn.Linear(1280, 3840, bias=True).to(
            dtype=b.current_dtype, device=b.device
        )
        linear.weight.data = torch.randn_like(linear.weight.data)
        linear.bias.data = torch.randn_like(linear.bias.data)
        return linear(x)

    return op


@register_op("Linear GPT2 FFN Up (1280->5120)", "model_linear")
def _linear_gpt2_ffn_up(b, size):
    def op():
        torch.manual_seed(42)
        x = b.create_test_tensor((8192, 1280))
        linear = torch.nn.Linear(1280, 5120, bias=True).to(
            dtype=b.current_dtype, device=b.device
        )
        linear.weight.data = torch.randn_like(linear.weight.data)
        linear.bias.data = torch.randn_like(linear.bias.data)
        return linear(x)

    return op


@register_op("Linear GPT2 FFN Down (5120->1280)", "model_linear")
def _linear_gpt2_ffn_down(b, size):
    def op():
        torch.manual_seed(42)
        x = b.create_test_tensor((8192, 5120))
        linear = torch.nn.Linear(5120, 1280, bias=True).to(
            dtype=b.current_dtype, device=b.device
        )
        linear.weight.data = torch.randn_like(linear.weight.data)
        linear.bias.data = torch.randn_like(linear.bias.data)
        return linear(x)

    return op


@register_op("Linear LLaMA SwiGLU Up (1200->11008)", "model_linear")
def _linear_llama_swiglu_up(b, size):
    def op():
        torch.manual_seed(42)
        x = b.create_test_tensor((8192, 1200))
        linear = torch.nn.Linear(1200, 11008, bias=False).to(
            dtype=b.current_dtype, device=b.device
        )
        linear.weight.data = torch.randn_like(linear.weight.data)
        return linear(x)

    return op


@register_op("Linear LLaMA SwiGLU Down (11008->1200)", "model_linear")
def _linear_llama_swiglu_down(b, size):
    def op():
        torch.manual_seed(42)
        x = b.create_test_tensor((8192, 11008))
        linear = torch.nn.Linear(11008, 1200, bias=False).to(
            dtype=b.current_dtype, device=b.device
        )
        linear.weight.data = torch.randn_like(linear.weight.data)
        return linear(x)

    return op


@register_op("Linear LLaMA QKV Proj (1200->1200)", "model_linear")
def _linear_llama_qkv(b, size):
    def op():
        torch.manual_seed(42)
        x = b.create_test_tensor((8192, 1200))
        linear = torch.nn.Linear(1200, 1200, bias=False).to(
            dtype=b.current_dtype, device=b.device
        )
        linear.weight.data = torch.randn_like(linear.weight.data)
        return linear(x)

    return op


@register_op("Linear LSTM Gates (1024->4096)", "model_linear")
def _linear_lstm_gates(b, size):
    def op():
        torch.manual_seed(42)
        x = b.create_test_tensor((32, 1024))
        linear = torch.nn.Linear(1024, 4096, bias=True).to(
            dtype=b.current_dtype, device=b.device
        )
        linear.weight.data = torch.randn_like(linear.weight.data)
        linear.bias.data = torch.randn_like(linear.bias.data)
        return linear(x)

    return op


@register_op("Linear AlexNet FC (9216->4096)", "model_linear")
def _linear_alexnet_fc(b, size):
    def op():
        torch.manual_seed(42)
        x = b.create_test_tensor((32, 9216))
        linear = torch.nn.Linear(9216, 4096, bias=True).to(
            dtype=b.current_dtype, device=b.device
        )
        linear.weight.data = torch.randn_like(linear.weight.data)
        linear.bias.data = torch.randn_like(linear.bias.data)
        return linear(x)

    return op


# ---------------------------------------------------------------------------
# Model-specific activation ops
# ---------------------------------------------------------------------------


@register_op("GELU New (GPT-2 tanh approx)", "model_activation")
def _gelu_new_op(b, size):
    def op():
        a = b.create_test_tensor((8192, 5120))
        # gelu_new: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return torch.nn.functional.gelu(a, approximate="tanh")

    return op


@register_op("SiLU Large (LLaMA MLP)", "model_activation")
def _silu_large_op(b, size):
    def op():
        a = b.create_test_tensor((8192, 11008))
        return torch.nn.functional.silu(a)

    return op


@register_op("Sigmoid", "model_activation")
def _sigmoid_model_op(b, size):
    def op():
        a = b.create_test_tensor(size)
        return torch.sigmoid(a)

    return op


# ---------------------------------------------------------------------------
# RMSNorm op (LLaMA-specific)
# ---------------------------------------------------------------------------


@register_op("RMSNorm Manual (LLaMA)", "model_normalization")
def _rmsnorm_manual_op(b, size):
    def op():
        torch.manual_seed(42)
        x = b.create_test_tensor((32, 256, 1200))
        weight = torch.randn(1200, device=b.device, dtype=b.current_dtype)
        eps = 1e-6
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        return x_normed * weight

    return op


# ---------------------------------------------------------------------------
# RoPE op (LLaMA-specific)
# ---------------------------------------------------------------------------


@register_op("RoPE (Rotary Position Embedding)", "model_attention")
def _rope_op(b, size):
    def op():
        torch.manual_seed(42)
        batch, num_heads, seq_len, head_dim = 32, 12, 256, 100
        q = b.create_test_tensor((batch, num_heads, seq_len, head_dim))

        # Precompute frequencies
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, head_dim, 2, device=b.device, dtype=torch.float32)
                / head_dim
            )
        )
        t = torch.arange(seq_len, device=b.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype=b.current_dtype)
        cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
        sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)

        # Apply rotation
        q1 = q[..., : head_dim // 2]
        q2 = q[..., head_dim // 2 :]
        rotated = torch.cat((-q2, q1), dim=-1)
        return q * cos_cached + rotated * sin_cached

    return op


# ---------------------------------------------------------------------------
# Causal attention op
# ---------------------------------------------------------------------------


@register_op("Causal SDPA (GPT-2/LLaMA)", "model_attention")
def _causal_sdpa_op(b, size):
    def op():
        torch.manual_seed(42)
        seq_len = 256
        q = b.create_test_tensor((32, 20, seq_len, 64))
        k = b.create_test_tensor((32, 20, seq_len, 64))
        v = b.create_test_tensor((32, 20, seq_len, 64))
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    return op


@register_op("Attention QK^T (LLaMA shape)", "model_attention")
def _attn_qkt_llama_op(b, size):
    def op():
        torch.manual_seed(42)
        q = b.create_test_tensor((32, 12, 256, 100))
        k = b.create_test_tensor((32, 12, 256, 100))
        scale = 1.0 / (100.0**0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        return torch.softmax(scores, dim=-1)

    return op


@register_op("Attention QK^T (GPT-2 shape)", "model_attention")
def _attn_qkt_gpt2_op(b, size):
    def op():
        torch.manual_seed(42)
        q = b.create_test_tensor((32, 20, 256, 64))
        k = b.create_test_tensor((32, 20, 256, 64))
        scale = 1.0 / (64.0**0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # Causal mask
        mask = torch.triu(
            torch.ones(256, 256, device=b.device, dtype=torch.bool), diagonal=1
        )
        fill_val = -1e4 if b.current_dtype == torch.float16 else -1e9
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), fill_val)
        return torch.softmax(scores, dim=-1)

    return op


@register_op("Attention QK^T (BERT shape)", "model_attention")
def _attn_qkt_bert_op(b, size):
    def op():
        torch.manual_seed(42)
        q = b.create_test_tensor((32, 16, 256, 64))
        k = b.create_test_tensor((32, 16, 256, 64))
        scale = 1.0 / (64.0**0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        return torch.softmax(scores, dim=-1)

    return op


# ---------------------------------------------------------------------------
# LSTM-specific ops
# ---------------------------------------------------------------------------


@register_op("LSTM Gate Computation", "model_recurrent")
def _lstm_gate_op(b, size):
    def op():
        torch.manual_seed(42)
        x = b.create_test_tensor((32, 1024))
        h = b.create_test_tensor((32, 1024))
        w_ih = torch.randn(4096, 1024, device=b.device, dtype=b.current_dtype)
        w_hh = torch.randn(4096, 1024, device=b.device, dtype=b.current_dtype)
        b_ih = torch.randn(4096, device=b.device, dtype=b.current_dtype)
        b_hh = torch.randn(4096, device=b.device, dtype=b.current_dtype)

        gates = torch.mm(x, w_ih.t()) + b_ih + torch.mm(h, w_hh.t()) + b_hh
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        return torch.cat([i, f, g, o], dim=1)

    return op


@register_op("LSTM Cell State Update", "model_recurrent")
def _lstm_cell_update_op(b, size):
    def op():
        torch.manual_seed(42)
        # Simulate 64 sequential cell state updates
        c = torch.zeros(32, 1024, device=b.device, dtype=b.current_dtype)
        for _ in range(64):
            f_gate = torch.sigmoid(b.create_test_tensor((32, 1024)))
            i_gate = torch.sigmoid(b.create_test_tensor((32, 1024)))
            g_gate = torch.tanh(b.create_test_tensor((32, 1024)))
            c = f_gate * c + i_gate * g_gate
        h = torch.sigmoid(b.create_test_tensor((32, 1024))) * torch.tanh(c)
        return h

    return op


# ---------------------------------------------------------------------------
# Composite pipeline tests
# ---------------------------------------------------------------------------


@register_pipeline("Transformer MLP")
def _transformer_mlp_pipeline(b, size):
    def pipeline():
        x = b.create_test_tensor(size)
        w1 = b.create_test_tensor(size)
        w2 = b.create_test_tensor(size)
        x = torch.matmul(x, w1)
        x = torch.nn.functional.gelu(x)
        x = torch.matmul(x, w2)
        return x

    return pipeline


@register_pipeline("Attention Block")
def _attention_block_pipeline(b, size):
    def pipeline():
        batch_size = 2
        seq_len = min(size[0], 256)
        hidden = min(size[1], 256)
        num_heads = 4
        head_dim = hidden // num_heads

        x = b.create_test_tensor((batch_size, seq_len, hidden))

        torch.manual_seed(42)
        wq = torch.randn(hidden, hidden, device=b.device, dtype=b.current_dtype)
        wk = torch.randn(hidden, hidden, device=b.device, dtype=b.current_dtype)
        wv = torch.randn(hidden, hidden, device=b.device, dtype=b.current_dtype)
        wo = torch.randn(hidden, hidden, device=b.device, dtype=b.current_dtype)

        q = (
            torch.matmul(x, wq)
            .view(batch_size, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        k = (
            torch.matmul(x, wk)
            .view(batch_size, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        v = (
            torch.matmul(x, wv)
            .view(batch_size, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden)
        out = torch.matmul(attn, wo)
        return out

    return pipeline


@register_pipeline("Conv Block")
def _conv_block_pipeline(b, size):
    def pipeline():
        x = b.create_test_tensor((16, 64, 32, 32))
        conv = torch.nn.Conv2d(64, 128, 3, padding=1).to(
            dtype=b.current_dtype, device=b.device
        )
        bn = torch.nn.BatchNorm2d(128).to(dtype=b.current_dtype, device=b.device)
        bn.eval()
        torch.manual_seed(42)
        conv.weight.data = torch.randn_like(conv.weight.data)
        conv.bias.data = torch.randn_like(conv.bias.data)
        bn.weight.data = torch.randn_like(bn.weight.data)
        bn.bias.data = torch.randn_like(bn.bias.data)
        x = conv(x)
        x = bn(x)
        x = torch.relu(x)
        return x

    return pipeline


# ---------------------------------------------------------------------------
# Model-accurate composite pipeline tests
# ---------------------------------------------------------------------------
# These pipelines replicate the exact operator chains from cpbench models.
# Model-level SDC was detected but operator-level was not -- these
# pipelines test the chained patterns that may expose hardware faults
# only visible when operators interact.


@register_pipeline("BERT Transformer Layer")
def _bert_transformer_layer_pipeline(b, size):
    def pipeline():
        torch.manual_seed(42)
        batch, seq_len, hidden = 32, 256, 1024
        num_heads, head_dim = 16, 64

        x = b.create_test_tensor((batch, seq_len, hidden))

        # Self-attention with BERT shapes
        wq = torch.randn(hidden, hidden, device=b.device, dtype=b.current_dtype)
        wk = torch.randn(hidden, hidden, device=b.device, dtype=b.current_dtype)
        wv = torch.randn(hidden, hidden, device=b.device, dtype=b.current_dtype)
        wo = torch.randn(hidden, hidden, device=b.device, dtype=b.current_dtype)

        q = (
            torch.matmul(x, wq)
            .view(batch, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        k = (
            torch.matmul(x, wk)
            .view(batch, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        v = (
            torch.matmul(x, wv)
            .view(batch, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        attn_out = torch.matmul(attn, wo)

        # Post-attention LayerNorm + residual
        ln1 = torch.nn.LayerNorm(hidden).to(dtype=b.current_dtype, device=b.device)
        ln1.weight.data = torch.randn_like(ln1.weight.data)
        ln1.bias.data = torch.randn_like(ln1.bias.data)
        x = ln1(x + attn_out)

        # FFN: Linear(1024->4096) + GELU + Linear(4096->1024)
        w1 = torch.randn(hidden, 4096, device=b.device, dtype=b.current_dtype)
        b1 = torch.randn(4096, device=b.device, dtype=b.current_dtype)
        w2 = torch.randn(4096, hidden, device=b.device, dtype=b.current_dtype)
        b2 = torch.randn(hidden, device=b.device, dtype=b.current_dtype)
        ffn = torch.matmul(x, w1) + b1
        ffn = torch.nn.functional.gelu(ffn)
        ffn = torch.matmul(ffn, w2) + b2

        # Post-FFN LayerNorm + residual
        ln2 = torch.nn.LayerNorm(hidden).to(dtype=b.current_dtype, device=b.device)
        ln2.weight.data = torch.randn_like(ln2.weight.data)
        ln2.bias.data = torch.randn_like(ln2.bias.data)
        return ln2(x + ffn)

    return pipeline


@register_pipeline("GPT-2 Transformer Layer")
def _gpt2_transformer_layer_pipeline(b, size):
    def pipeline():
        torch.manual_seed(42)
        batch, seq_len, hidden = 32, 256, 1280
        num_heads, head_dim = 20, 64

        x = b.create_test_tensor((batch, seq_len, hidden))

        # Pre-norm (GPT-2 uses pre-LayerNorm)
        ln1 = torch.nn.LayerNorm(hidden).to(dtype=b.current_dtype, device=b.device)
        ln1.weight.data = torch.randn_like(ln1.weight.data)
        ln1.bias.data = torch.randn_like(ln1.bias.data)
        normed = ln1(x)

        # Fused QKV projection (1280 -> 3840)
        w_qkv = torch.randn(hidden, 3 * hidden, device=b.device, dtype=b.current_dtype)
        b_qkv = torch.randn(3 * hidden, device=b.device, dtype=b.current_dtype)
        qkv = torch.matmul(normed, w_qkv) + b_qkv
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

        # Causal attention
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, hidden)

        # Output projection
        wo = torch.randn(hidden, hidden, device=b.device, dtype=b.current_dtype)
        attn_out = torch.matmul(attn, wo)

        # Residual
        x = x + attn_out

        # Pre-norm for FFN
        ln2 = torch.nn.LayerNorm(hidden).to(dtype=b.current_dtype, device=b.device)
        ln2.weight.data = torch.randn_like(ln2.weight.data)
        ln2.bias.data = torch.randn_like(ln2.bias.data)
        normed2 = ln2(x)

        # FFN: Linear(1280->5120) + GELU_new + Linear(5120->1280)
        w1 = torch.randn(hidden, 4 * hidden, device=b.device, dtype=b.current_dtype)
        b1 = torch.randn(4 * hidden, device=b.device, dtype=b.current_dtype)
        w2 = torch.randn(4 * hidden, hidden, device=b.device, dtype=b.current_dtype)
        b2 = torch.randn(hidden, device=b.device, dtype=b.current_dtype)
        ffn = torch.matmul(normed2, w1) + b1
        ffn = torch.nn.functional.gelu(ffn, approximate="tanh")
        ffn = torch.matmul(ffn, w2) + b2

        return x + ffn

    return pipeline


@register_pipeline("LLaMA Transformer Layer")
def _llama_transformer_layer_pipeline(b, size):
    def pipeline():
        torch.manual_seed(42)
        batch, seq_len, hidden = 32, 256, 1200
        num_heads, head_dim = 12, 100
        intermediate = 11008
        eps = 1e-6

        x = b.create_test_tensor((batch, seq_len, hidden))

        # RMSNorm (pre-attention)
        w_rms1 = torch.randn(hidden, device=b.device, dtype=b.current_dtype)
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + eps) * w_rms1

        # Q/K/V projections (no bias)
        wq = torch.randn(hidden, hidden, device=b.device, dtype=b.current_dtype)
        wk = torch.randn(hidden, hidden, device=b.device, dtype=b.current_dtype)
        wv = torch.randn(hidden, hidden, device=b.device, dtype=b.current_dtype)

        q = (
            torch.matmul(normed, wq)
            .view(batch, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        k = (
            torch.matmul(normed, wk)
            .view(batch, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        v = (
            torch.matmul(normed, wv)
            .view(batch, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )

        # RoPE
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, head_dim, 2, device=b.device, dtype=torch.float32)
                / head_dim
            )
        )
        t = torch.arange(seq_len, device=b.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype=b.current_dtype)
        cos_e = emb.cos().unsqueeze(0).unsqueeze(0)
        sin_e = emb.sin().unsqueeze(0).unsqueeze(0)

        q1, q2 = q[..., : head_dim // 2], q[..., head_dim // 2 :]
        q = q * cos_e + torch.cat((-q2, q1), dim=-1) * sin_e
        k1, k2 = k[..., : head_dim // 2], k[..., head_dim // 2 :]
        k = k * cos_e + torch.cat((-k2, k1), dim=-1) * sin_e

        # Causal attention
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, hidden)

        # O projection
        wo = torch.randn(hidden, hidden, device=b.device, dtype=b.current_dtype)
        attn_out = torch.matmul(attn, wo)

        # Residual
        x = x + attn_out

        # RMSNorm (pre-MLP)
        w_rms2 = torch.randn(hidden, device=b.device, dtype=b.current_dtype)
        variance2 = x.pow(2).mean(-1, keepdim=True)
        normed2 = x * torch.rsqrt(variance2 + eps) * w_rms2

        # SwiGLU MLP
        w_gate = torch.randn(
            hidden, intermediate, device=b.device, dtype=b.current_dtype
        )
        w_up = torch.randn(hidden, intermediate, device=b.device, dtype=b.current_dtype)
        w_down = torch.randn(
            intermediate, hidden, device=b.device, dtype=b.current_dtype
        )

        gate = torch.nn.functional.silu(torch.matmul(normed2, w_gate))
        up = torch.matmul(normed2, w_up)
        mlp_out = torch.matmul(gate * up, w_down)

        return x + mlp_out

    return pipeline


@register_pipeline("LLaMA SwiGLU MLP")
def _llama_swiglu_pipeline(b, size):
    def pipeline():
        torch.manual_seed(42)
        hidden, intermediate = 1200, 11008
        x = b.create_test_tensor((8192, hidden))

        w_gate = torch.randn(
            hidden, intermediate, device=b.device, dtype=b.current_dtype
        )
        w_up = torch.randn(hidden, intermediate, device=b.device, dtype=b.current_dtype)
        w_down = torch.randn(
            intermediate, hidden, device=b.device, dtype=b.current_dtype
        )

        gate = torch.nn.functional.silu(torch.matmul(x, w_gate))
        up = torch.matmul(x, w_up)
        return torch.matmul(gate * up, w_down)

    return pipeline


@register_pipeline("GPT-2 FFN Block")
def _gpt2_ffn_pipeline(b, size):
    def pipeline():
        torch.manual_seed(42)
        hidden = 1280
        x = b.create_test_tensor((8192, hidden))

        w1 = torch.randn(hidden, 4 * hidden, device=b.device, dtype=b.current_dtype)
        b1 = torch.randn(4 * hidden, device=b.device, dtype=b.current_dtype)
        w2 = torch.randn(4 * hidden, hidden, device=b.device, dtype=b.current_dtype)
        b2 = torch.randn(hidden, device=b.device, dtype=b.current_dtype)

        out = torch.matmul(x, w1) + b1
        out = torch.nn.functional.gelu(out, approximate="tanh")
        out = torch.matmul(out, w2) + b2
        return out

    return pipeline


@register_pipeline("BERT FFN Block")
def _bert_ffn_pipeline(b, size):
    def pipeline():
        torch.manual_seed(42)
        hidden, intermediate = 1024, 4096
        x = b.create_test_tensor((8192, hidden))

        w1 = torch.randn(hidden, intermediate, device=b.device, dtype=b.current_dtype)
        b1 = torch.randn(intermediate, device=b.device, dtype=b.current_dtype)
        w2 = torch.randn(intermediate, hidden, device=b.device, dtype=b.current_dtype)
        b2 = torch.randn(hidden, device=b.device, dtype=b.current_dtype)

        out = torch.matmul(x, w1) + b1
        out = torch.nn.functional.gelu(out)
        out = torch.matmul(out, w2) + b2
        return out

    return pipeline


@register_pipeline("LSTM Multi-Step Gates")
def _lstm_multistep_pipeline(b, size):
    def pipeline():
        torch.manual_seed(42)
        batch_sz, hidden = 32, 1024
        w_ih = torch.randn(4096, 1024, device=b.device, dtype=b.current_dtype)
        w_hh = torch.randn(4096, 1024, device=b.device, dtype=b.current_dtype)
        b_ih = torch.randn(4096, device=b.device, dtype=b.current_dtype)
        b_hh = torch.randn(4096, device=b.device, dtype=b.current_dtype)

        h = torch.zeros(batch_sz, hidden, device=b.device, dtype=b.current_dtype)
        c = torch.zeros(batch_sz, hidden, device=b.device, dtype=b.current_dtype)

        # Run 32 timesteps to test sequential accumulation
        for step in range(32):  # noqa: B007
            x = b.create_test_tensor((batch_sz, hidden))
            gates = torch.mm(x, w_ih.t()) + b_ih + torch.mm(h, w_hh.t()) + b_hh
            i, f, g, o = gates.chunk(4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)

        return h

    return pipeline


@register_pipeline("Multi-Layer Residual Chain")
def _multi_layer_residual_pipeline(b, size):
    def pipeline():
        torch.manual_seed(42)
        hidden = 1024
        x = b.create_test_tensor((8192, hidden))

        # Simulate 12 residual layers (like BERT)
        for layer_idx in range(12):  # noqa: B007
            w = torch.randn(hidden, hidden, device=b.device, dtype=b.current_dtype)
            bias = torch.randn(hidden, device=b.device, dtype=b.current_dtype)
            residual = x
            x = torch.matmul(x, w) + bias
            x = torch.nn.functional.gelu(x)
            x = x + residual  # residual connection

        return x

    return pipeline


# ---------------------------------------------------------------------------
# Backward pass operations — gradient computation exercises different CUDA
# kernels than forward pass and is where training SDC often manifests.
# ---------------------------------------------------------------------------


@register_op("Backward Linear (BERT FFN)", "backward_pass")
def _backward_linear_bert(b, size):
    def op():
        torch.manual_seed(42)
        batch = 8192
        x = torch.randn(
            batch, 1024, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        w = torch.randn(
            4096, 1024, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        y = torch.nn.functional.linear(x, w)
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        # Return concatenation of both gradients for bit-exact comparison
        return torch.cat([x.grad.flatten(), w.grad.flatten()]).cpu()

    return op


@register_op("Backward Linear (LLaMA SwiGLU Up)", "backward_pass")
def _backward_linear_llama_up(b, size):
    def op():
        torch.manual_seed(42)
        batch = 8192
        x = torch.randn(
            batch, 1200, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        w = torch.randn(
            11008, 1200, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        y = torch.nn.functional.linear(x, w)
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        return torch.cat([x.grad.flatten(), w.grad.flatten()]).cpu()

    return op


@register_op("Backward GELU", "backward_pass")
def _backward_gelu(b, size):
    def op():
        torch.manual_seed(42)
        x = torch.randn(
            8192, 4096, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        y = torch.nn.functional.gelu(x)
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        return x.grad.cpu()

    return op


@register_op("Backward GELU New (tanh approx)", "backward_pass")
def _backward_gelu_new(b, size):
    def op():
        torch.manual_seed(42)
        x = torch.randn(
            8192, 5120, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        y = torch.nn.functional.gelu(x, approximate="tanh")
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        return x.grad.cpu()

    return op


@register_op("Backward SiLU", "backward_pass")
def _backward_silu(b, size):
    def op():
        torch.manual_seed(42)
        x = torch.randn(
            8192, 11008, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        y = torch.nn.functional.silu(x)
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        return x.grad.cpu()

    return op


@register_op("Backward LayerNorm", "backward_pass")
def _backward_layernorm(b, size):
    def op():
        torch.manual_seed(42)
        x = torch.randn(
            32, 256, 1024, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        ln = torch.nn.LayerNorm(1024, device=b.device, dtype=b.current_dtype)
        y = ln(x)
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        return x.grad.cpu()

    return op


@register_op("Backward RMSNorm", "backward_pass")
def _backward_rmsnorm(b, size):
    def op():
        torch.manual_seed(42)
        x = torch.randn(
            32, 256, 1200, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        weight = torch.randn(
            1200, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        eps = 1e-5
        variance = x.pow(2).mean(-1, keepdim=True)
        y = x * torch.rsqrt(variance + eps) * weight
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        return torch.cat([x.grad.flatten(), weight.grad.flatten()]).cpu()

    return op


@register_op("Backward Softmax", "backward_pass")
def _backward_softmax(b, size):
    def op():
        torch.manual_seed(42)
        x = torch.randn(
            32, 16, 256, 256, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        y = torch.nn.functional.softmax(x, dim=-1)
        grad_out = torch.randn_like(y)
        y.backward(grad_out)
        return x.grad.cpu()

    return op


@register_op("Backward Matmul (QK^T)", "backward_pass")
def _backward_matmul_qkt(b, size):
    def op():
        torch.manual_seed(42)
        # Simulates attention score backward: Q @ K^T
        q = torch.randn(
            32, 12, 256, 100, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        k = torch.randn(
            32, 12, 256, 100, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        scores = torch.matmul(q, k.transpose(-2, -1)) / (100**0.5)
        grad_out = torch.randn_like(scores)
        scores.backward(grad_out)
        return torch.cat([q.grad.flatten(), k.grad.flatten()]).cpu()

    return op


@register_op("Backward Attention Values (scores @ V)", "backward_pass")
def _backward_attn_values(b, size):
    def op():
        torch.manual_seed(42)
        # Simulates attention output backward: softmax(scores) @ V
        attn_weights = torch.randn(
            32, 12, 256, 256, device=b.device, dtype=b.current_dtype
        )
        attn_weights = torch.softmax(attn_weights, dim=-1).detach().requires_grad_(True)
        v = torch.randn(
            32, 12, 256, 100, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        out = torch.matmul(attn_weights, v)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        return torch.cat([attn_weights.grad.flatten(), v.grad.flatten()]).cpu()

    return op


@register_op("Backward Cross Entropy Loss", "backward_pass")
def _backward_cross_entropy(b, size):
    def op():
        torch.manual_seed(42)
        # Simulates LM head backward (logits -> loss)
        logits = torch.randn(
            8192, 32000, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        targets = torch.randint(0, 32000, (8192,), device=b.device)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        return logits.grad.cpu()

    return op


# ---------------------------------------------------------------------------
# Training-mode pipelines — forward + backward + optimizer step.
# These match what model-level SDC tests actually do.
# ---------------------------------------------------------------------------


@register_pipeline("Training Step: BERT FFN Layer")
def _training_bert_ffn(b, size):
    def pipeline():
        num_steps = int(os.environ.get("CROSSGPU_TRAINING_STEPS", "4"))
        torch.manual_seed(42)
        batch, hidden, ffn_dim = 256, 1024, 4096

        # Create "parameters"
        w1 = torch.randn(
            ffn_dim, hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        b1 = torch.randn(
            ffn_dim, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        w2 = torch.randn(
            hidden, ffn_dim, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        b2 = torch.randn(
            hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        ln_w = torch.ones(
            hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        ln_b = torch.zeros(
            hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )

        params = [w1, b1, w2, b2, ln_w, ln_b]
        optimizer = torch.optim.Adam(params, lr=1e-4)

        # Run training steps
        all_grads = []
        for step in range(num_steps):
            torch.manual_seed(42 + step)
            x = torch.randn(batch, hidden, device=b.device, dtype=b.current_dtype)
            target = torch.randn(batch, hidden, device=b.device, dtype=b.current_dtype)

            optimizer.zero_grad()
            # LayerNorm
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + 1e-5)
            x_ln = x_norm * ln_w + ln_b
            # FFN: up -> gelu -> down
            h = torch.nn.functional.linear(x_ln, w1, b1)
            h = torch.nn.functional.gelu(h)
            out = torch.nn.functional.linear(h, w2, b2)
            # Residual + loss
            out = out + x
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            if num_steps <= 10:
                all_grads.append(torch.cat([p.grad.flatten() for p in params]))
            optimizer.step()

        if num_steps > 10:
            return torch.cat([p.flatten() for p in params]).cpu()
        return torch.cat(all_grads).cpu()

    return pipeline


@register_pipeline("Training Step: LLaMA SwiGLU MLP")
def _training_llama_mlp(b, size):
    def pipeline():
        num_steps = int(os.environ.get("CROSSGPU_TRAINING_STEPS", "4"))
        torch.manual_seed(42)
        batch, hidden, intermediate = 256, 1200, 11008

        gate_proj = torch.randn(
            intermediate,
            hidden,
            device=b.device,
            dtype=b.current_dtype,
            requires_grad=True,
        )
        up_proj = torch.randn(
            intermediate,
            hidden,
            device=b.device,
            dtype=b.current_dtype,
            requires_grad=True,
        )
        down_proj = torch.randn(
            hidden,
            intermediate,
            device=b.device,
            dtype=b.current_dtype,
            requires_grad=True,
        )
        rms_w = torch.ones(
            hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )

        params = [gate_proj, up_proj, down_proj, rms_w]
        optimizer = torch.optim.Adam(params, lr=1e-4)

        all_grads = []
        for step in range(num_steps):
            torch.manual_seed(42 + step)
            x = torch.randn(batch, hidden, device=b.device, dtype=b.current_dtype)
            target = torch.randn(batch, hidden, device=b.device, dtype=b.current_dtype)

            optimizer.zero_grad()
            # RMSNorm
            variance = x.pow(2).mean(-1, keepdim=True)
            x_norm = x * torch.rsqrt(variance + 1e-5) * rms_w
            # SwiGLU
            gate = torch.nn.functional.silu(
                torch.nn.functional.linear(x_norm, gate_proj)
            )
            up = torch.nn.functional.linear(x_norm, up_proj)
            out = torch.nn.functional.linear(gate * up, down_proj)
            # Residual + loss
            out = out + x
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            if num_steps <= 10:
                all_grads.append(torch.cat([p.grad.flatten() for p in params]))
            optimizer.step()

        if num_steps > 10:
            # Stress mode: return final parameter state (accumulated corruption)
            return torch.cat([p.flatten() for p in params]).cpu()
        return torch.cat(all_grads).cpu()

    return pipeline


@register_pipeline("Training Step: Full Attention Block")
def _training_attention(b, size):
    def pipeline():
        num_steps = int(os.environ.get("CROSSGPU_TRAINING_STEPS", "4"))
        torch.manual_seed(42)
        batch, seq, hidden, heads = 32, 128, 1024, 16
        head_dim = hidden // heads

        wq = torch.randn(
            hidden, hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        wk = torch.randn(
            hidden, hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        wv = torch.randn(
            hidden, hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        wo = torch.randn(
            hidden, hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        ln_w = torch.ones(
            hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )

        params = [wq, wk, wv, wo, ln_w]
        optimizer = torch.optim.Adam(params, lr=1e-4)

        all_grads = []
        for step in range(num_steps):
            torch.manual_seed(42 + step)
            x = torch.randn(batch, seq, hidden, device=b.device, dtype=b.current_dtype)
            target = torch.randn(
                batch, seq, hidden, device=b.device, dtype=b.current_dtype
            )

            optimizer.zero_grad()
            # RMSNorm
            variance = x.pow(2).mean(-1, keepdim=True)
            x_norm = x * torch.rsqrt(variance + 1e-5) * ln_w
            # Q, K, V projections
            q = (x_norm @ wq.t()).view(batch, seq, heads, head_dim).transpose(1, 2)
            k = (x_norm @ wk.t()).view(batch, seq, heads, head_dim).transpose(1, 2)
            v = (x_norm @ wv.t()).view(batch, seq, heads, head_dim).transpose(1, 2)
            # Attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
            # Causal mask
            mask = torch.triu(torch.ones(seq, seq, device=b.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(batch, seq, hidden)
            out = out @ wo.t()
            # Residual + loss
            out = out + x
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            if num_steps <= 10:
                all_grads.append(torch.cat([p.grad.flatten() for p in params]))
            optimizer.step()

        if num_steps > 10:
            return torch.cat([p.flatten() for p in params]).cpu()
        return torch.cat(all_grads).cpu()

    return pipeline


@register_pipeline("Training Step: GPT-2 Transformer Layer")
def _training_gpt2_layer(b, size):
    def pipeline():
        num_steps = int(os.environ.get("CROSSGPU_TRAINING_STEPS", "4"))
        torch.manual_seed(42)
        batch, seq, hidden = 16, 128, 1280
        heads, head_dim = 20, 64
        ffn_dim = 5120

        # Attention params
        w_qkv = torch.randn(
            3 * hidden,
            hidden,
            device=b.device,
            dtype=b.current_dtype,
            requires_grad=True,
        )
        w_out = torch.randn(
            hidden, hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        ln1_w = torch.ones(
            hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        ln1_b = torch.zeros(
            hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        # FFN params
        w_fc = torch.randn(
            ffn_dim, hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        b_fc = torch.randn(
            ffn_dim, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        w_proj = torch.randn(
            hidden, ffn_dim, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        b_proj = torch.randn(
            hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        ln2_w = torch.ones(
            hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )
        ln2_b = torch.zeros(
            hidden, device=b.device, dtype=b.current_dtype, requires_grad=True
        )

        params = [w_qkv, w_out, ln1_w, ln1_b, w_fc, b_fc, w_proj, b_proj, ln2_w, ln2_b]
        optimizer = torch.optim.AdamW(params, lr=1e-4)

        all_grads = []
        for step in range(num_steps):
            torch.manual_seed(42 + step)
            x = torch.randn(batch, seq, hidden, device=b.device, dtype=b.current_dtype)
            target = torch.randn(
                batch, seq, hidden, device=b.device, dtype=b.current_dtype
            )

            optimizer.zero_grad()
            residual = x
            # Pre-LayerNorm
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True, unbiased=False)
            x_n = (x - mean) / torch.sqrt(var + 1e-5) * ln1_w + ln1_b
            # Fused QKV
            qkv = torch.nn.functional.linear(x_n, w_qkv)
            q, k, v = qkv.split(hidden, dim=-1)
            q = q.view(batch, seq, heads, head_dim).transpose(1, 2)
            k = k.view(batch, seq, heads, head_dim).transpose(1, 2)
            v = v.view(batch, seq, heads, head_dim).transpose(1, 2)
            # Causal attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
            mask = torch.triu(torch.ones(seq, seq, device=b.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, hidden)
            attn_out = torch.nn.functional.linear(attn_out, w_out)
            x = residual + attn_out
            # FFN with pre-LayerNorm
            residual2 = x
            mean2 = x.mean(-1, keepdim=True)
            var2 = x.var(-1, keepdim=True, unbiased=False)
            x_n2 = (x - mean2) / torch.sqrt(var2 + 1e-5) * ln2_w + ln2_b
            h = torch.nn.functional.linear(x_n2, w_fc, b_fc)
            h = torch.nn.functional.gelu(h, approximate="tanh")
            out = torch.nn.functional.linear(h, w_proj, b_proj)
            out = residual2 + out
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            if num_steps <= 10:
                all_grads.append(torch.cat([p.grad.flatten() for p in params]))
            optimizer.step()

        if num_steps > 10:
            return torch.cat([p.flatten() for p in params]).cpu()
        return torch.cat(all_grads).cpu()

    return pipeline


# ---------------------------------------------------------------------------
# Per-operator training pipelines — isolate individual GeMM operations.
# When a pipeline fails, the name tells you exactly which operator is corrupt.
# Use with --concurrent --num-workers 8 --training-steps 100 to trigger SDC.
# ---------------------------------------------------------------------------


def _single_linear_training_pipeline(name, in_features, out_features, batch=256):
    """Factory for single Linear operator training pipelines."""

    @register_pipeline(f"Training Step: {name} ({in_features}x{out_features})")
    def _pipeline(b, size):
        def pipeline():
            num_steps = int(os.environ.get("CROSSGPU_TRAINING_STEPS", "4"))
            torch.manual_seed(42)

            weight = torch.randn(
                out_features,
                in_features,
                device=b.device,
                dtype=b.current_dtype,
                requires_grad=True,
            )
            bias = torch.randn(
                out_features,
                device=b.device,
                dtype=b.current_dtype,
                requires_grad=True,
            )

            params = [weight, bias]
            optimizer = torch.optim.Adam(params, lr=1e-5, foreach=True)

            all_grads = []
            for step in range(num_steps):
                torch.manual_seed(42 + step)
                x = torch.randn(
                    batch, in_features, device=b.device, dtype=b.current_dtype
                )
                target = torch.randn(
                    batch, out_features, device=b.device, dtype=b.current_dtype
                )

                optimizer.zero_grad()
                out = torch.nn.functional.linear(x, weight, bias)
                loss = torch.nn.functional.mse_loss(out, target)
                loss.backward()
                if num_steps <= 10:
                    all_grads.append(torch.cat([p.grad.flatten() for p in params]))
                optimizer.step()

            if num_steps > 10:
                return torch.cat([p.flatten() for p in params]).cpu()
            return torch.cat(all_grads).cpu()

        return pipeline

    return _pipeline


# LLaMA SwiGLU MLP individual projections (matches cpbench pytorch-llama-large)
_single_linear_training_pipeline("gate_proj", 1200, 11008)
_single_linear_training_pipeline("up_proj", 1200, 11008)
_single_linear_training_pipeline("down_proj", 11008, 1200)

# LLaMA attention projections
_single_linear_training_pipeline("q_proj", 1200, 1200)
_single_linear_training_pipeline("k_proj", 1200, 1200)
_single_linear_training_pipeline("v_proj", 1200, 1200)
_single_linear_training_pipeline("o_proj", 1200, 1200)

# Classification head
_single_linear_training_pipeline("linear_head", 1200, 100)
