## 1. Create the Environment

```bash
conda env create -f environment.yml
conda activate agentretro-core
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## 2. Prepare Reaction Data

The input data must contain at least `reaction_id`, `product`, `reactants`, and `split`. MKA additionally requires `retro_template`, `minimal_template`, and `scaffold`.

`chemistry.py` handles SMILES normalization, template extraction, molecular fingerprints, scaffolds, and Top-k matching.

## 3. Build SDE Domains

Call `sde.fit_sde_frame(frame)` to:

1. Compute Morgan fingerprints for products and reaction differences.
2. Build exact Tanimoto kNN graphs.
3. Perform spectral embedding and capacity-constrained k-means clustering.
4. Create 12 real training domains and write `domain_id` to the returned table.

## 4. Generate MKA Data and Text Features

Run the components in `mka.py` in the following order:

1. Use `deterministic_chembl_sample` to draw a deterministic sample from ChEMBL.
2. Use `screen_candidates` to apply property and similarity filters.
3. Use `VLLMEngine` with Qwen to generate candidate products. It falls back to Transformers when vLLM is unavailable.
4. Use `generate_pseudo_ood` to apply training templates and construct pseudo-OOD reactions. Set their `domain_id` to `0`.
5. Use `generate_training_rationales` to generate reaction text.
6. Use `encode_rationales` with SciBERT to obtain 768-dimensional `text_embedding` vectors.

## 5. Connect a Backbone

The shared interface is defined in `backbones/base.py`. Available adapters are:

- `EditRetroAdapter`
- `RetroComposerAdapter`
- `GDiffRetroAdapter`

## 6. Train and Run Inference

Create the training settings with `TrainSettings`, then wrap the backbone adapter and optimizer with `BilevelTrainer`. Each training batch must contain the backbone-specific inputs, `domain_id`, and `text_embedding`.

```python
from agentretro.training import BilevelTrainer, TrainSettings

settings = TrainSettings()
trainer = BilevelTrainer(adapter, optimizer, settings)
metrics = trainer.train_step(batch)
predictions = adapter.predict(inference_batch, top_k=10)
```

Training applies the task loss, domain-classifier update, GRL adversarial learning, and the MLDG bilevel update. Text information used during training is disabled at inference time.
