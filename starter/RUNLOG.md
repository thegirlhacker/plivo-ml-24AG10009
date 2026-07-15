# Run Log

## Run 0: Baseline Model
* **Hypothesis**: The baseline is mediocre because raw byte tokenization splits Hindi (Devanagari) characters into 3 separate tokens, exhausting the context window of 128 tokens extremely fast. Additionally, constant learning rate (3e-4) without warmup/decay, a small batch size (8), and untied embedding weights waste parameter budget and limit learning.
* **What changed**: Used the default mediocre starter codebase: raw byte tokenizer, 4 layers, 160 embedding dimension, 4 heads, constant LR, no weight decay, no gradient clipping.
* **Tied parameters**: 1,339,840
* **Checkpoint parameters**: 1,339,840
* **Dev BPB**: 2.3718
* **Conclusion**: Establishing a baseline of 2.3718 BPB. Byte-level tokenization is highly inefficient for Devanagari text. A BPE tokenizer is required to group bytes and expand the effective context window.

## Run 1: Modernized 7-Layer RoPE Model (BPE Vocab 4096)
* **Hypothesis**: Replacing the raw byte tokenizer with a 4096-vocabulary BPE tokenizer will compress Devanagari characters into single tokens, reducing token sequence length and expanding the effective context window by ~2.4x. Implementing Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU MLP, and weight tying will maximize parameter efficiency, allowing us to train a deeper 7-layer model under the 2.0M parameter cap. Using a Cosine LR schedule with linear warmup (peak LR 1.5e-3) and AdamW with weight decay will accelerate convergence and stabilize training.
* **What changed**: 
  - Tokenizer: Byte-Pair Encoding (BPE) with vocab size 4096 trained on the corpus.
  - Positional Encodings: Replaced absolute positional embeddings with Rotary Positional Embeddings (RoPE).
  - Normalization: Replaced LayerNorm with RMSNorm.
  - Activation: Replaced GELU in MLP with SwiGLU.
  - Parameters: Enabled weight tying and removed bias parameters in linear projections.
  - Optimizer & Scheduler: AdamW with weight decay (0.1 on weights, 0.0 on norms/embeddings) + Cosine LR schedule (100 steps warmup, 2000 steps total, peak LR 1.5e-3, min LR 1.5e-4) + Gradient clipping (1.0).
  - Batch size: Increased batch size from 8 to 32.
  - Checkpoint stripping: Overrode `load_state_dict` in `model.py` and stripped `"head.weight"` from the saved checkpoint in `train.py` to guarantee parameter deduplication in the checkpoint file itself.
* **Tied parameters**: 1,901,568
* **Checkpoint parameters**: 1,901,568 (head weight stripped from state dict, dynamically re-tied on load)
* **Dev BPB**: 1.7320
* **Conclusion**: Massive success! Dev BPB dropped from 2.3718 to 1.7320. The BPE tokenizer achieved a compression ratio of 2.41 bytes/token. The deeper 7-layer model trained smoothly and safely under the parameter and step limits.
