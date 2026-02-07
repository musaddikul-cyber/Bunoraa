"""
Fine-tune a seq2seq model (T5) to generate structured JSON product suggestions.
Expected training data: JSONL with fields: {"text": "...", "target": "{...}"}

Usage:
python apps/products/ml/train_product_suggestor.py --data data/products_suggest_train.jsonl --model t5-small --out_dir ./outputs/suggestor_v1

Requires: transformers, datasets, torch
"""
from pathlib import Path
import argparse
import json
import torch
import numpy as np

# Optional imports guarded
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
except Exception:
    raise RuntimeError("Missing ML dependencies. Install with: pip install -r requirements-ml.txt")


def compute_metrics(p):
    # NA for generative JSON outputs, user should implement task-specific metrics if desired
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', default='t5-small')
    parser.add_argument('--out_dir', default='./outputs/product_suggestor')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset('json', data_files={'train': args.data, 'validation': args.data})

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess(examples):
        inputs = examples['text']
        model_inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=256)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target'], truncation=True, padding='max_length', max_length=256)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized = ds.map(preprocess, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_strategy='epoch',
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(str(out_dir / 'hf_model'))
    tokenizer.save_pretrained(str(out_dir / 'hf_model'))

    # Save simple label mapping (none for generation)
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump({'model': args.model}, f)

    print('Training finished. Export with export_product_suggestor.py')


if __name__ == '__main__':
    main()