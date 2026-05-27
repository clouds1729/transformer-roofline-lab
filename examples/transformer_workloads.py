"""Transformer workload examples for the FRAME roofline demo."""

from __future__ import annotations

from src.transformer_roofline import TransformerWorkload


TRANSFORMER_WORKLOADS = {
    "tiny_transformer": TransformerWorkload(
        name="tiny_transformer",
        batch_size=1,
        sequence_length=128,
        hidden_dim=256,
        num_layers=4,
        num_attention_heads=4,
        mlp_expansion_ratio=4.0,
        bytes_per_element=2,
        notes="Small classroom/debug workload using bf16/fp16-sized elements.",
    ),
    "bert_base_like": TransformerWorkload(
        name="bert_base_like",
        batch_size=1,
        sequence_length=512,
        hidden_dim=768,
        num_layers=12,
        num_attention_heads=12,
        mlp_expansion_ratio=4.0,
        bytes_per_element=2,
        notes="BERT-base-shaped encoder-style workload; illustrative only.",
    ),
    "gpt2_small_like": TransformerWorkload(
        name="gpt2_small_like",
        batch_size=1,
        sequence_length=1024,
        hidden_dim=768,
        num_layers=12,
        num_attention_heads=12,
        mlp_expansion_ratio=4.0,
        bytes_per_element=2,
        notes="GPT-2-small-shaped decoder block stack; KV-cache effects omitted.",
    ),
}


def get_transformer_workloads() -> list[TransformerWorkload]:
    return list(TRANSFORMER_WORKLOADS.values())
