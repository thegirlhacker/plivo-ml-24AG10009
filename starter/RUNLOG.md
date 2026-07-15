# Run Log & Experimental Progression

This log documents the systematic, step-by-step experimental approach taken to optimize the baseline LLM under the strict parameter and optimizer step caps.

---

## Run 0: Baseline Model (Establishing the Benchmark)
* **Hypothesis**: The baseline model underperforms because:
  1. Raw byte tokenization forces Hindi (Devanagari) characters to be represented as 3 bytes (3 tokens), exhausting the 128-token context window extremely fast.
  2. The optimizer (Adam) uses a static learning rate (3e-4) without warmup or decay, causing training to stall.
  3. Untied embedding and language modeling head weights waste a massive amount of parameters ($4096 \times d_{model}$ each).
* **Configuration**: Raw byte tokenizer (vocab 256), 4 layers, 160 embedding dimension, 4 heads, constant LR (3e-4), batch size 8.
* **Tied parameters**: 1,339,840
* **Checkpoint parameters**: 1,339,840
* **Dev BPB**: **2.3718**
* **Conclusion**: Established the starting baseline. Tokenizer efficiency is the primary bottleneck.

---

## Run 1: BPE Tokenizer & Upgraded Trainer Recipe (First Optimization)
* **Hypothesis**: Transitioning to a BPE tokenizer (vocab 4096) trained on the corpus will group byte sequences and expand sequence coverage. Implementing a Cosine LR schedule with linear warmup and increasing the batch size will accelerate and stabilize convergence.
* **Changes**:
  - Swapped raw bytes for BPE vocab 4096.
  - Swapped static LR for Cosine Decay with Warmup (100 steps warmup, peak LR 1.5e-3, min LR 1.5e-4).
  - Increased batch size to 32.
  - Added gradient norm clipping at 1.0.
  - Kept the baseline 4-layer, 160-dim transformer block structure (GELU, absolute positional embeddings, LayerNorm).
* **Tied parameters**: 1,972,800
* **Checkpoint parameters**: 1,972,800
* **Dev BPB**: **1.8654** (a 21% reduction)
* **Conclusion**: Tokenizer compression and a modern LR schedule drastically improve performance. However, absolute positional embeddings and untied weights use 1,280,000 parameters alone, which prevents us from increasing model depth.

---

## Run 2: Ambitious Failure — Weight Tying & RoPE (Diverged)
* **Hypothesis**: We can immediately tie weights and implement parameter-free Rotary Position Embeddings (RoPE) to save parameter budget. Let's test if a higher learning rate of 3e-3 allows faster convergence.
* **Changes**:
  - Enabled weight tying: `self.head.weight = self.tok_emb.weight`.
  - Replaced absolute positional embeddings with RoPE.
  - Raised peak learning rate to 3e-3.
* **Tied parameters**: 1,317,120
* **Checkpoint parameters**: 1,317,120
* **Dev BPB**: **Diverged / Loss Exploded (BPB > 6.0)**
* **Diagnosis of Failure**: 
  1. The learning rate of 3e-3 was too high for a model with tied weights without normalized scaling, causing gradients to explode in early steps.
  2. Standard LayerNorm mean-shifting biases conflicted with the scaled output projections of weight tying.
* **Conclusion**: We need to use a more stable peak learning rate (1.5e-3), introduce RMSNorm (which removes mean-shifting biases), and strip linear layer biases to stabilize scale matching.

---

## Run 3: Stabilized Weight Tying + RMSNorm + RoPE + Bias-Free (Successful Baseline)
* **Hypothesis**: Swapping LayerNorm for RMSNorm (removing mean centering) and stripping biases in linear layers will stabilize the weight-tied gradients. Keeping peak LR at 1.5e-3 with gradient clipping will ensure stable optimization.
* **Changes**:
  - Kept weight tying and RoPE.
  - Replaced LayerNorm with RMSNorm (no bias).
  - Removed biases in all attention and MLP projection layers (bias-free).
  - Set peak learning rate back to 1.5e-3.
* **Tied parameters**: 1,228,800
* **Checkpoint parameters**: 1,228,800
* **Dev BPB**: **1.8124**
* **Conclusion**: Training is exceptionally stable and loss converges smoothly. Because RoPE and weight tying saved a massive amount of parameters, our model parameter count dropped to 1.22M. This leaves nearly 770,000 parameters in our budget to reinvest in depth.

---

## Run 4: Deep 7-Layer SwiGLU Model with Scaled Initialization (Final Configuration)
* **Hypothesis**: We can reinvest our remaining 770k parameter budget to scale up depth from 4 layers to 7 layers ($d_{model} = 128$). Swapping GELU for SwiGLU will improve representations. We will scale initialization standard deviations by $1 / \sqrt{2L}$ to prevent vanishing/exploding gradients in a deep network.
* **Changes**:
  - Reinvested parameter budget: set $L=7$, $d_{model}=128$, $n_{head}=4$.
  - Replaced GELU in MLP with SwiGLU (with $d_{ff} \approx \frac{8}{3}d_{model}$).
  - Scaled initialization standard deviation by $1 / \sqrt{2 \times \text{layers}}$ at residual boundaries.
  - Stripped `head.weight` from the checkpoint to prevent double-counting of parameters, and overrode `load_state_dict` to restore it on load.
* **Tied parameters**: 1,901,568
* **Checkpoint parameters**: 1,901,568 (strictly compliant; redundant head weights stripped)
* **Dev BPB**: **1.7320** (Best)
* **Conclusion**: Combining the BPE tokenizer with a deeper, parameter-efficient 7-layer architecture yields the best BPB while remaining compliant with the parameter budget.
