import torch

from toto.model.fusion import Fusion
from toto.model.transformer import Transformer
from toto.model.util import KVCache


def _make_transformer(embed_dim: int, num_heads: int, use_mea: bool = False) -> Transformer:
    """
    Create a minimal Transformer with one spacewise layer followed by one timewise layer.
    Set use_memory_efficient_attention to False in tests to avoid xformers dependency.
    """
    return Transformer(
        num_layers=2,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_hidden_dim=embed_dim * 2,
        dropout=0.0,
        spacewise_every_n_layers=2,
        spacewise_first=True,
        use_memory_efficient_attention=use_mea,
        fusion=Fusion(),
    )


def _make_variate_label_embeds(
    batch: int, variate: int, embed_dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.randn(batch, variate, 1, embed_dim, device=device, dtype=dtype)


@torch.no_grad()
def test_fusion_prepends_token_without_kvcache():
    device = torch.device("cpu")
    dtype = torch.float32
    batch, variate, seq_len, embed_dim, heads = 2, 3, 4, 8, 2

    transformer = _make_transformer(embed_dim, heads, use_mea=False).to(device=device, dtype=dtype)

    embeddings = torch.randn(batch, variate, seq_len, embed_dim, device=device, dtype=dtype)
    id_mask = torch.zeros(batch, variate, seq_len, device=device, dtype=dtype)
    variate_label_embeds = _make_variate_label_embeds(batch, variate, embed_dim, device, dtype)

    out = transformer(embeddings, id_mask, kv_cache=None, variate_label_embeds=variate_label_embeds)

    # One token per variate should be prepended to the sequence dimension
    assert out.shape == (batch, variate, seq_len + 1, embed_dim)


@torch.no_grad()
def test_fusion_prepends_only_initial_step_with_kvcache():
    device = torch.device("cpu")
    dtype = torch.float32
    batch, variate, seq_len, embed_dim, heads = 2, 3, 4, 8, 2

    transformer = _make_transformer(embed_dim, heads, use_mea=False).to(device=device, dtype=dtype)

    # Fresh KV cache (length 0)
    kv_cache = KVCache(
        batch_size=batch,
        num_variates=variate,
        transformer_layers=list(transformer.layers),
        num_layers=len(transformer.layers),
        embed_dim=embed_dim,
        num_heads=heads,
        max_seq_len=32,
        device=device,
        dtype=dtype,
        use_memory_efficient_attention=False,
    )

    embeddings = torch.randn(batch, variate, seq_len, embed_dim, device=device, dtype=dtype)
    id_mask = torch.zeros(batch, variate, seq_len, device=device, dtype=dtype)
    variate_label_embeds = _make_variate_label_embeds(batch, variate, embed_dim, device, dtype)

    # Initial forward with empty cache -> fusion applied (prepend 1)
    out1 = transformer(embeddings, id_mask, kv_cache=kv_cache, variate_label_embeds=variate_label_embeds)
    assert out1.shape == (batch, variate, seq_len + 1, embed_dim)
    # KV cache should now reflect the new total time length on timewise layer
    assert kv_cache.current_len(0) == seq_len + 1

    # Next step: provide one-step inputs (typical decoding) -> fusion skipped
    step_embeddings = torch.randn(batch, variate, 1, embed_dim, device=device, dtype=dtype)
    step_id_mask = torch.zeros(batch, variate, 1, device=device, dtype=dtype)
    out2 = transformer(step_embeddings, step_id_mask, kv_cache=kv_cache, variate_label_embeds=variate_label_embeds)
    assert out2.shape == (batch, variate, 1, embed_dim)
    # Cache length should have increased by 1 without extra fusion token
    assert kv_cache.current_len(0) == seq_len + 2
