# Run Log & Experimental Progression

This log documents the systematic, step-by-step experimental approach taken to optimize the baseline LLM under the strict parameter and optimizer step caps.

---

## Run 0: Baseline Model (Establishing the Benchmark)
* **Hypothesis**: The baseline model is highly suboptimal because:
  1. Raw byte tokenization forces Devanagari (Hindi) characters to be represented as 3 bytes (3 tokens), exhausting the 128-token context window extremely fast.
  2. The optimizer (Adam) uses a static learning rate (3e-4) without warmup or decay, causing training to stall.
  3. Untied embedding and language modeling head weights waste a massive amount of parameters ($4096 \times d_{model}$ each).
* **Configuration**: Raw byte tokenizer (vocab 256), 4 layers, 160 embedding dimension, 4 heads, constant LR (3e-4), batch size 8.
* **Tied parameters**: 1,339,840
* **Checkpoint parameters**: 1,339,840
* **Dev BPB**: **2.3718**
* **Conclusion**: Established a baseline BPB of 2.3718. The main bottleneck is token sequence length. We must transition to a BPE tokenizer to group character bytes.

---

## Run 1: BPE Tokenizer + Upgraded Trainer Recipe (First Optimization)
* **Hypothesis**: Transitioning to a BPE tokenizer (vocab 4096) will compress natural text by over 2x, extending the effective context window. Introducing Cosine Learning Rate Decay with linear warmup and increasing the batch size will accelerate convergence.
* **What changed**:
  - Swapped raw bytes for BPE vocab 4096 (merges trained on the training corpus).
  - Swapped static LR for Cosine Decay with Warmup (100 steps warmup, peak LR 1.5e-3, min LR 1.5e-4).
  - Increased batch size from 8 to 32 (training on 4x more tokens per step).
  - Added gradient norm clipping at 1.0.
  - Kept the baseline 4-layer, 160-dim transformer block structure (GELU, absolute positional embeddings, LayerNorm).
* **Tied parameters**: 1,972,800
* **Checkpoint parameters**: 1,972,800
* **Dev BPB**: **1.8654** (a huge 21% reduction)
* **Conclusion**: Tokenizer compression and a modern LR schedule drastically improve performance. However, absolute positional embeddings and untied weights use 1,280,000 parameters alone, which prevents us from increasing model depth.

---

## Run 2: Architecture Swap — Weight Tying, RoPE, RMSNorm & SwiGLU (Final Configuration)
* **Hypothesis**: By implementing weight tying and parameter-free Rotary Position Embeddings (RoPE), we can eliminate absolute position tables and output head weights. This allows us to re-invest the saved parameter budget into building a deeper model (increasing from 4 to 7 layers) to improve hierarchical abstraction. Swapping LayerNorm for RMSNorm and GELU for SwiGLU will further optimize training speed and expressivity.
* **What changed**:
  - Enabled weight tying: `self.head.weight = self.tok_emb.weight`.
  - Replaced absolute positional embeddings with Rotary Positional Embeddings (RoPE).
  - Replaced LayerNorm with RMSNorm (saving biases).
  - Replaced GELU in MLP with SwiGLU (with $d_{ff} \approx \frac{8}{3}d_{model}$).
  - Increased depth from 4 layers ($d_{model} = 160$) to 7 layers ($d_{model} = 128$).
  - Added custom PyTorch state-dict interception: stripped `head.weight` from the saved checkpoint in `train.py` and overrode `load_state_dict` in `model.py` to restore the tie on load.
* **Tied parameters**: 1,901,568
* **Checkpoint parameters**: 1,901,568 (strictly compliant; redundant head weights stripped)
* **Dev BPB**: **1.7320** (a further 7% reduction, 27% total improvement over baseline)
* **Conclusion**: Combining BPE tokenizer compression with a deep, parameter-efficient 7-layer architecture yields the best bits-per-byte performance while remaining strictly within all hard caps.
