"""Generate a 1,000-sample SEO-friendly JSONL dataset.

This script uses curated templates and concise, benefit-driven copy informed by
best practices (e.g., Yoast Shopify guidance) and synthesizes original text.

Run:
  python apps/products/ml/generate_seo_dataset.py --count 1000 --seed 52

Output:
  apps/products/ml/datasets/seo_products_1000.jsonl

Note: Content is synthesized and original; it is not copied from any site.
"""
from pathlib import Path
import json
import argparse
import random
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('generate_seo_dataset')

OUT_DIR = Path(__file__).resolve().parent / 'datasets'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLUG_RE = re.compile(r'[^a-z0-9-]+')

TEMPLATES = [
    {
        'base_name': 'Premium Leather Wallet',
        'category': 'accessories',
        'features': ['Full-grain leather', 'RFID-blocking', '6 card slots', 'Compact fit'],
        'weight_range': (0.08, 0.18),
    },
    {
        'base_name': 'Handmade Ceramic Vase',
        'category': 'home',
        'features': ['Wheel-thrown ceramic', 'Matte glaze', 'Gift-ready packaging'],
        'weight_range': (0.9, 2.4),
    },
    {
        'base_name': 'Organic Cotton T-Shirt',
        'category': 'fashion',
        'features': ['GOTS-certified cotton', 'Breathable knit', 'True-to-size fit'],
        'weight_range': (0.15, 0.28),
    },
    {
        'base_name': 'Solid Oak Dining Table',
        'category': 'furniture',
        'features': ['Solid oak top', 'Hand-sanded finish', 'Seating for 6'],
        'weight_range': (25.0, 45.0),
    },
    {
        'base_name': 'Insulated Steel Bottle',
        'category': 'kitchen',
        'features': ['Double-wall insulation', '316 stainless steel', 'Leak-proof cap'],
        'weight_range': (0.25, 0.7),
    },
    {
        'base_name': 'Merino Wool Scarf',
        'category': 'fashion',
        'features': ['Fine merino yarn', 'Thermo-regulating', 'Hypoallergenic'],
        'weight_range': (0.12, 0.4),
    },
    {
        'base_name': 'Silk Blend Shawl',
        'category': 'luxury',
        'features': ['Hand-finished edges', 'Lightweight drape', 'Elegant sheen'],
        'weight_range': (0.06, 0.22),
    },
    {
        'base_name': 'Crystal-Glass Mug',
        'category': 'kitchen',
        'features': ['Crystal-clear glass', 'Dishwasher safe', 'Comfort grip handle'],
        'weight_range': (0.18, 0.5),
    },
]

ADJECTIVES = ['Handcrafted', 'Signature', 'Refined', 'Heritage', 'Modern', 'Eco']


def slugify(s: str, idx: int) -> str:
    v = s.lower().replace(' ', '-').replace('#', '')
    v = SLUG_RE.sub('', v)
    return f"{v}-{idx}"


def make_premium(idx: int) -> dict:
    rng = random.Random(idx)
    template = TEMPLATES[idx % len(TEMPLATES)]
    adj = ADJECTIVES[idx % len(ADJECTIVES)]
    base = template['base_name']
    name = f"{adj} {base}"
    slug = slugify(f"{adj} {base}", idx)
    sku = f"BUN{idx:06d}"

    # Build a premium description: short intro, benefits, features, care
    intro = f"{template['category'].title()} crafted for lasting beauty and everyday use."
    benefits = (
        f"Designed for comfort and durability, this {base.lower()} offers superior materials, careful construction, and an elevated finish."
    )
    features = ' • '.join(template['features'])
    care = "Care: spot clean or gentle cycle; avoid harsh detergents (see full care guide)."

    description = ' '.join([intro, benefits, 'Features:', features + '.', care])
    short_description = f"{adj} {base} with premium materials and refined design."

    low_w, high_w = template['weight_range']
    frac = ((idx * 19) % 100) / 100.0
    weight = round(low_w + (high_w - low_w) * frac, 2)

    stock_quantity = max(5, 100 - (idx % 200))
    low_stock_threshold = 3 + (idx % 5)

    keyword = base  # Primary keyword for SEO
    meta_title = f"Buy {adj} {base} | Premium {template['category'].title()} - Bunoraa"
    meta_title = meta_title[:70]
    meta_description = (f"Shop {name}. {short_description} Free shipping over $50. Limited stock available.")
    if len(meta_description) > 155:
        meta_description = meta_description[:152].rstrip() + '…'

    tags = list(dict.fromkeys(template['features'][:2] + [template['category'], 'premium']))

    return {
        'name': name,
        'slug': slug,
        'sku': sku,
        'description': description,
        'short_description': short_description,
        'stock_quantity': stock_quantity,
        'low_stock_threshold': low_stock_threshold,
        'tags': tags,
        'weight': weight,
        'meta_title': meta_title,
        'meta_description': meta_description,
        'meta_keywords': ', '.join([keyword, template['category'], 'premium']),
    }


def main(count: int = 1000, seed: int = 52):
    random.seed(seed)
    out_file = OUT_DIR / f'seo_products_{count}.jsonl'
    logger.info('Generating %s samples -> %s', count, out_file)

    with out_file.open('w', encoding='utf-8') as fh:
        for i in range(1, count + 1):
            rec = make_premium(i)
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if i % 200 == 0:
                logger.info('Wrote %s samples...', i)

    logger.info('Done. Wrote %s samples to %s', count, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=52)
    args = parser.parse_args()
    main(count=args.count, seed=args.seed)
