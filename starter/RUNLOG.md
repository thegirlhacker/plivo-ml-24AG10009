# Run Log

## Run 0: Baseline
* **Hypothesis**: The baseline model underperforms due to raw byte tokenization (splitting Hindi characters into 3 separate tokens and limiting context size) and a suboptimal optimizer setup (constant learning rate without warmup or decay).
* **Changes**: Baseline configuration (raw byte tokenizer, 4 layers, 160 dimension, 4 heads, constant LR 3e-4, batch size 8).
* **Parameters**: 1,339,840
* **Dev BPB**: 2.3718
* **Conclusion**: Established the starting baseline. Tokenizer efficiency is the primary bottleneck.

## Run 1: BPE Tokenizer & Upgraded Trainer
* **Hypothesis**: Transitioning to a BPE tokenizer (vocab 4096) trained on the corpus will group byte sequences and expand sequence coverage. Implementing a Cosine LR schedule with linear warmup and increasing the batch size will accelerate and stabilize convergence.
* **Changes**: 
  - Integrated 4096 BPE vocabulary.
  - Implemented Cosine LR decay (100 steps warmup, peak LR 1.5e-3, min LR 1.5e-4).
  - Increased batch size to 32.
  - Added gradient norm clipping at 1.0.
* **Parameters**: 1,972,800
* **Dev BPB**: 1.8654
* **Conclusion**: BPE tokenization and the updated trainer recipe significantly reduced BPB. However, the absolute position tables and untied output weights consume too much parameter budget.

## Run 2: RoPE, Weight Tying, and SwiGLU Architecture (Final)
* **Hypothesis**: Implementing weight tying and parameter-free RoPE positional encodings will free up enough parameter budget to increase depth from 4 to 7 layers. Swapping LayerNorm for RMSNorm and GELU for SwiGLU will improve representation capacity.
* **Changes**:
  - Tied embedding and output head weights.
  - Replaced absolute positional embeddings with Rotary Positional Embeddings (RoPE).
  - Replaced LayerNorm with RMSNorm and GELU with SwiGLU.
  - Increased depth from 4 layers ($d_{model} = 160$) to 7 layers ($d_{model} = 128$).
  - Stripped `head.weight` from the saved checkpoint to prevent double-counting of parameters, and overrode `load_state_dict` to restore it on load.
* **Parameters**: 1,901,568
* **Dev BPB**: 1.7320
* **Conclusion**: Combining the BPE tokenizer with a deeper, parameter-efficient 7-layer architecture yields the best BPB while remaining compliant with the parameter budget.
