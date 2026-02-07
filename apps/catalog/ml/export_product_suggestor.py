"""
Wrap a trained HF seq2seq model as a `ProductSuggestor` and save as `product_suggestor.pt`.

Usage:
python apps/products/ml/export_product_suggestor.py --hf_dir ./outputs/product_suggestor/hf_model --out_dir ./apps/products/ml/artifacts --jit
"""
import argparse
from pathlib import Path
import json
import torch
from typing import List

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:
    AutoTokenizer = AutoModelForSeq2SeqLM = None


class ProductSuggestor(torch.nn.Module):
    def __init__(self, model, tokenizer, device='cpu'):
        super().__init__()
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer

    def predict(self, texts: List[str], max_length: int = 128) -> List[dict]:
        results = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i+16]
            enc = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_length=max_length, num_return_sequences=1)
            dec = [self.tokenizer.decode(g, skip_special_tokens=True) for g in out]
            for d in dec:
                # try parse JSON
                try:
                    parsed = json.loads(d)
                except Exception:
                    parsed = {'raw': d}
                results.append(parsed)
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()

    hf_dir = Path(args.hf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        raise RuntimeError("Missing 'transformers' dependency. Install with: pip install -r requirements-ml.txt or pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(hf_dir))

    wrapper = ProductSuggestor(model, tokenizer, device=args.device)
    artifact = out_dir / 'product_suggestor.pt'
    torch.save({'model': wrapper, 'meta': {}}, artifact)
    print(f'Saved product suggestor to {artifact}')

    if args.jit:
        example = "Handwoven shawl in silk"
        enc = tokenizer([example], return_tensors='pt', padding=True, truncation=True)
        scripted = torch.jit.trace(wrapper, (enc['input_ids'].to(args.device),))
        torch.jit.save(scripted, out_dir / 'product_suggestor_jit.pt')


if __name__ == '__main__':
    main()