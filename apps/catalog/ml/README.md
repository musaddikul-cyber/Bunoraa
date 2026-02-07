Product Suggestor ML tools

This folder contains training, export and dataset preparation tools for the product suggestor.

- `train_product_suggestor.py` — T5 seq2seq training script.
- `export_product_suggestor.py` — Export wrapper that saves `product_suggestor.pt` under `ml/artifacts/` by default.
- `prepare_large_dataset.py` — Ingestion and Arrow-shard writer with optional S3 upload. It now generates a `manifest.json` in the output directory describing shards (row counts, sizes, files) and will upload the manifest to S3 alongside the shards when `--s3-output` is provided.

Artifacts
---------
Saved artifacts are placed by default in `apps/products/ml/artifacts/` (used by the running app loader).
