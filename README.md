# Nystorm vs Simple Transformer

Nystrom Attention applies the nystorm attention to the matrix multiplicaiton of Self-Attention 
with landmarks chosen using Recursive Randomized Ridge Leverage Scores.

---


## Current bugs

1. Loss Issue where some gradients go to Nan. To see full Gradients, uncomment lines  158-164 in RRLS_Nystrom/train_translate_nystrom
Inf Gradients:
encoder_embedding.lut.weight: inf
encoder.transformer_blocks.0.multi_head_attention.attention_blocks.0.query_embed.weight: inf
encoder.transformer_blocks.0.multi_head_attention.attention_blocks.0.query_embed.bias: inf
encoder.transformer_blocks.0.multi_head_attention.attention_blocks.0.key_embed.weight: inf


2. Divide by zero warning before first epoch. I believe this is due to num_landmarks not being initialized and being defaulted to 0.

Extra Info:
Hyperparameters to tune:
- rrls_lambda (RRLS_Nystrom/Transformer/sub_layers_nystrom.py)
- rrls_gamma (RRLS_Nystrom/Transformer/sub_layers_nystrom.py)
- learning_rate (RRLS_Nystrom/train_translate_nystrom.py)
- Gradient clipping max norm (RRLS_Nystrom/train_translate_nystrom.py)
- num_landmarks (RRLS_Nystrom/train_traintranslate_nystrom.py)

---

### Install Requirements
```
cd RRLS_Nystrom
python -m pip install -r requirements.txt
```

### Run Traning Script
Nystrom Attention Transformer
```
python train_translate_nystorm.py
```

### Plot Loss for training checkpoint.pkl
Note: Will need to modify direcotries for checkpoint models
`RRLS_Nystrom/Extra_Scripts/plot_loss.ipynb`


### Credits
Simple Transformer Implementaiton: https://github.com/IpsumDominum/Pytorch-Simple-Transformer
Nystromformer: https://arxiv.org/pdf/2102.03902
RRLS: https://arxiv.org/abs/1605.07583
