# Multimodel Architecture

```bash
MultiModalModel(
  (models): ModuleList(
    (0): MultiheadAttention(
      (embedding): Linear(in_features=76, out_features=128, bias=True)
      (trans): TransformerEncoder(
        (layers): ModuleList(
          (0-7): 8 x TransformerEncoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
            )
            (linear1): Linear(in_features=128, out_features=2048, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
            (linear2): Linear(in_features=2048, out_features=128, bias=True)
            (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (dropout1): Dropout(p=0.1, inplace=False)
            (dropout2): Dropout(p=0.1, inplace=False)
            (activation): GELU(approximate='none')
          )
        )
      )
      (pos_embedding): PositionalEncoding(
        (dropout): Dropout(p=0, inplace=False)
      )
    )
    (1): BERT(
      (model): BertModel(
        (embeddings): BertEmbeddings(
          (word_embeddings): Embedding(28996, 128, padding_idx=0)
          (position_embeddings): Embedding(512, 128)
          (token_type_embeddings): Embedding(2, 128)
          (LayerNorm): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): BertEncoder(
          (layer): ModuleList(
            (0-7): 8 x BertLayer(
              (attention): BertAttention(
                (self): BertSdpaSelfAttention(
                  (query): Linear(in_features=128, out_features=128, bias=True)
                  (key): Linear(in_features=128, out_features=128, bias=True)
                  (value): Linear(in_features=128, out_features=128, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=128, out_features=128, bias=True)
                  (LayerNorm): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=128, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=128, bias=True)
                (LayerNorm): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (pooler): BertPooler(
          (dense): Linear(in_features=128, out_features=128, bias=True)
          (activation): Tanh()
        )
      )
    )
  )
  (projector): Identity()
  (clip_crit): CLIP_Loss()
  (mse_crit1): SmoothL1Loss()
  (mse_crit2): NLLLoss()
  (final_layer1): Linear(in_features=128, out_features=76, bias=True)
  (final_layer2): Linear(in_features=128, out_features=28996, bias=True)
  (softmax): LogSoftmax(dim=-1)
  (kl_div): KLDivLoss()
)
```

# Training log

```bash
wandb: Run summary:
wandb:         Average Measurement Tau 0.2
wandb:               Average Notes Tau 0.2
wandb:                           Epoch 30
wandb:                       Eval Loss 2.57355
wandb:            Eval Zeroshot AUC-PR 0.21965
wandb:           Eval Zeroshot AUC-ROC 0.68349
wandb: Eval Zeroshot Balanced Accuracy 0.59811
wandb:                       Test Loss 3.09532
wandb:            Test Zeroshot AUC-PR 0.21965
wandb:           Test Zeroshot AUC-ROC 0.68349
wandb: Test Zeroshot Balanced Accuracy 0.59811
wandb:                      Train Loss 7.82402
wandb:                         meas_r1 0.31492
wandb:                        meas_r10 2.76163
wandb:                         meas_r5 1.42926
wandb:                     meas_r_mean 1.50194
wandb:                          r_mean 1.5948
wandb:                          txt_r1 0.21802
wandb:                         txt_r10 3.17345
wandb:                          txt_r5 1.67151
wandb:                      txt_r_mean 1.68766
wandb: 
wandb: 🚀 View run exp at: https://wandb.ai/causal_language_trainer/Multimodal%20Clinical%20Pretraining/runs/sp9u8c9z
wandb: ⭐️ View project at: https://wandb.ai/causal_language_trainer/Multimodal%20Clinical%20Pretraining
wandb: Synced 7 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20250227_025605-sp9u8c9z/logs
```