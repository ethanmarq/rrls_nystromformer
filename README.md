# Nystorm vs Simple Transformer

Nystrom Attention applies the nystorm attention to the matrix multiplicaiton of Self-Attention 
with landmarks chosen using Recursive Randomized Ridge Leverage Scores.

---


## Current bugs
Loss does not decrease, hence it is not learning. However loss is no longer Nan and RRLS Nystrom
is run rather than Standard Attention.

My presumption is to tune hyperparameters and remove or adjust gradient clipping.
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
