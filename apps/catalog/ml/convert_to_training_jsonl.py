"""Convert a product dataset JSONL into training JSONL compatible with train_product_suggestor.py

Each input record must have: name, short_description, description, tags
Output records contain:
 - text: short_description + " \n\n" + description
 - target: JSON string with keys: name, short_description, tags

Usage:
  python apps/products/ml/convert_to_training_jsonl.py --in apps/products/ml/datasets/seo_products_1000_model_generated.jsonl --out apps/products/ml/datasets/train_1000.jsonl
"""
from pathlib import Path
import json
import argparse


def convert(input_path: Path, output_path: Path):
    with input_path.open('r', encoding='utf-8') as inf, output_path.open('w', encoding='utf-8') as outf:
        for line in inf:
            obj = json.loads(line)
            text = (obj.get('short_description', '') or '') + '\n\n' + (obj.get('description', '') or '')
            target_dict = {
                'name': obj.get('name', ''),
                'short_description': obj.get('short_description', ''),
                'tags': obj.get('tags', []),
            }
            rec = {'text': text, 'target': json.dumps(target_dict, ensure_ascii=False)}
            outf.write(json.dumps(rec, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='inpath', required=True)
    parser.add_argument('--out', dest='outpath', required=True)
    args = parser.parse_args()
    convert(Path(args.inpath), Path(args.outpath))
