# Attention Is All You Need

PyTorch implementation of the Transformer from [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

This is a from-scratch encoder-decoder Transformer trained on the [opus_books](https://huggingface.co/datasets/opus_books) English-Italian dataset.

## Project Structure

- `model.py` — Transformer architecture: embeddings, positional encoding, multi-head attention, encoder/decoder blocks, and projection layer.
- `dataset.py` — `BilingualDataset` for source/target tokenization, padding, and causal masking.
- `train.py` — Training loop, validation with greedy decoding, and TensorBoard logging.
- `config.py` — Hyperparameters and checkpoint paths.

## Training

```bash
python train.py
```

Checkpoints are saved to `opus_books_weights/` and TensorBoard logs to `runs/`.

## Dependencies

- PyTorch
- Hugging Face `datasets`
- Hugging Face `tokenizers`
- `torchmetrics`
- `tqdm`
- `tensorboard`
