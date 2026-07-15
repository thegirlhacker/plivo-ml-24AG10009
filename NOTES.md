1. Our best configuration is a 7-layer transformer model with 128 embedding dimensions, 4 attention heads, and a 4,096-vocabulary Byte-Pair Encoding (BPE) tokenizer.
2. Weight tying between the input embedding and output head projection matrices saves 524,288 parameters, enabling a deeper 7-layer architecture under the 2,000,000 parameter budget.
3. The custom BPE tokenizer trained on the corpus compresses Hindi and English text by 2.41x (bytes per token), effectively expanding the model's sequence coverage from 128 to over 300 characters.
4. Rotary Position Embeddings (RoPE) are used because they are parameter-free and capture relative distances better than absolute positional embeddings.
5. RMSNorm is chosen over LayerNorm because it operates faster and saves parameters by eliminating the mean-centering bias terms.
6. The SwiGLU activation function is implemented in the MLP block to improve model capacity and convergence speed.
7. Biases in the linear projections are removed to prevent overfitting and optimize the parameter count.
8. Training is stabilized using an AdamW optimizer with a weight decay of 0.1 applied selectively to 2D weight matrices.
9. A cosine learning rate schedule with linear warmup (peak LR 1.5e-3, min LR 1.5e-4) allows the model to converge rapidly and escape local minima.
10. The checkpoint is saved by stripping the tied head weight, ensuring that parameter counting directly from the file is strictly compliant at 1,901,568 parameters.
