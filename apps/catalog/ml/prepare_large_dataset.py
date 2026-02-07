"""
Prepare large datasets for product suggestor training.
Supports streaming JSONL from local file or S3 (s3://bucket/path/*.jsonl) and validates per-row JSON schema.
Outputs an Arrow dataset directory usable by HuggingFace `datasets` loader.

Usage examples:
  python apps/products/ml/prepare_large_dataset.py --input s3://my-bucket/path/train-*.jsonl --out_dir ./data/arrow --max-shards 100
  python apps/products/ml/prepare_large_dataset.py --input ./data/raw/train.jsonl --out_dir ./data/arrow --validate

Notes:
- Requires `datasets` and `s3fs` for S3 streaming.
- For very large data, run on a machine with sufficient disk to hold Arrow shards or write direct to S3-backed Arrow if desired.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable

import jsonschema

# Optional datasets guard
try:
    from datasets import load_dataset, Dataset, DatasetDict, Features, Value
    from datasets import Dataset as HFDataset
except Exception:
    load_dataset = None
    HFDataset = None

# Optional boto3 for S3 uploads
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:
    boto3 = None
    BotoCoreError = ClientError = Exception

logger = logging.getLogger('prepare_dataset')
logging.basicConfig(level=logging.INFO)

SCHEMA = Path(__file__).resolve().parent.parent / 'schemas' / 'product_suggest.schema.json'


def iter_jsonl(uri: str) -> Iterable[dict]:
    """Yield JSON objects from a local file or s3 URL using datasets streaming if available."""
    if uri.startswith('s3://') and load_dataset is not None:
        # datasets can stream from s3 if s3fs is installed
        ds = load_dataset('json', data_files={'data': uri}, streaming=True)['data']
        for rec in ds:
            yield rec
        return
    # local file
    p = Path(uri)
    if not p.exists():
        raise FileNotFoundError(uri)
    with p.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning('Skipping invalid JSON line')
                continue


def validate_row(row: dict) -> bool:
    schema = json.loads(SCHEMA.read_text(encoding='utf-8'))
    try:
        jsonschema.validate(instance=row, schema=schema)
        return True
    except jsonschema.ValidationError as exc:
        logger.debug('Validation failed: %s', exc)
        return False


def write_arrow(out_dir: Path, rows: Iterable[dict], shard_size: int = 100000):
    """Consume rows generator and write Arrow shards to out_dir using datasets.

    Returns a list of shard metadata dicts: [{"shard_name": ..., "row_count": ...}, ...]
    """
    if HFDataset is None:
        raise RuntimeError('datasets not available. Install with: pip install datasets[torch]')
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = []
    count = 0
    shard = 0
    shards = []
    for r in rows:
        buf.append(r)
        count += 1
        if len(buf) >= shard_size:
            ds = HFDataset.from_list(buf)
            shard_dir = out_dir / f'shard_{shard}'
            ds.save_to_disk(str(shard_dir))
            logger.info('Wrote shard %s with %s rows', shard, len(buf))
            shards.append({"shard_name": f'shard_{shard}', "row_count": len(buf), "path": shard_dir.as_posix()})
            shard += 1
            buf = []
    if buf:
        ds = HFDataset.from_list(buf)
        shard_dir = out_dir / f'shard_{shard}'
        ds.save_to_disk(str(shard_dir))
        logger.info('Wrote shard %s with %s rows', shard, len(buf))
        shards.append({"shard_name": f'shard_{shard}', "row_count": len(buf), "path": shard_dir.as_posix()})
    return shards


def upload_directory_to_s3(directory: Path, bucket: str, prefix: str = ''):
    """Upload files from a directory to S3 preserving relative paths.

    directory: local path with shard_* subdirectories
    bucket: s3 bucket name
    prefix: optional key prefix in the bucket (no leading/trailing slash)
    """
    if boto3 is None:
        raise RuntimeError('boto3 not installed. Install with `pip install boto3` to upload to S3.')

    client = boto3.client('s3')
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(base)

    for root, _, files in os.walk(base):
        for f in files:
            local_path = Path(root) / f
            rel = local_path.relative_to(base)
            key = f"{prefix.rstrip('/')}/{rel.as_posix()}" if prefix else rel.as_posix()
            logger.info('Uploading %s -> s3://%s/%s', local_path, bucket, key)
            try:
                client.upload_file(str(local_path), bucket, key)
            except (BotoCoreError, ClientError) as exc:
                logger.warning('Failed to upload %s: %s', local_path, exc)
                raise


def create_manifest(out_dir: Path, shards_meta: list, manifest_name: str = 'manifest.json') -> Path:
    """Create a manifest.json in out_dir describing the generated shards.

    shards_meta: list of dicts with at least keys 'shard_name' and 'row_count'
    Returns Path to manifest file.
    """
    import time
    manifest = {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'total_shards': len(shards_meta),
        'total_rows': sum(s.get('row_count', 0) for s in shards_meta),
        'shards': [],
    }
    base = Path(out_dir)
    for s in shards_meta:
        shard_dir = base / s.get('shard_name')
        size = 0
        mtime = 0
        files = []
        if shard_dir.exists():
            for root, _, filenames in os.walk(shard_dir):
                for fn in filenames:
                    p = Path(root) / fn
                    try:
                        st = p.stat()
                        size += st.st_size
                        mtime = max(mtime, int(st.st_mtime))
                        files.append(str(p.relative_to(base).as_posix()))
                    except FileNotFoundError:
                        continue
        manifest['shards'].append({
            'shard_name': s.get('shard_name'),
            'row_count': s.get('row_count', 0),
            'size_bytes': size,
            'mtime': mtime,
            'files': files,
        })
    manifest_path = base / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    logger.info('Wrote manifest %s', manifest_path)
    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input JSONL path (local or s3://)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--validate', action='store_true', help='Validate rows against JSON Schema')
    parser.add_argument('--shard_size', type=int, default=100000)
    parser.add_argument('--max_shards', type=int, default=0, help='Stop after this many shards (0 = unlimited)')
    parser.add_argument('--s3-output', type=str, help='Optional s3://bucket/prefix path to upload shards to after processing')
    parser.add_argument('--s3-remove-local', action='store_true', help='Remove local shards after successful upload')
    args = parser.parse_args()

    if args.validate and not SCHEMA.exists():
        raise RuntimeError(f'Schema not found: {SCHEMA}')

    rows = []
    total = 0
    it = iter_jsonl(args.input)
    def gen():
        nonlocal total
        for row in it:
            if args.validate and not validate_row(row):
                continue
            total += 1
            yield row
    out_dir = Path(args.out_dir)
    shards_meta = write_arrow(out_dir, gen(), shard_size=args.shard_size)
    logger.info('Total rows written: %s', total)

    # write manifest describing the shards
    manifest_path = create_manifest(out_dir, shards_meta)

    # Optional upload to S3
    if args.s3_output:
        if boto3 is None:
            raise RuntimeError('boto3 is required to upload shards to S3. Install with `pip install boto3`')
        # s3 path should be s3://bucket/prefix
        s3 = args.s3_output
        if not s3.startswith('s3://'):
            raise RuntimeError('s3 output must be of the form s3://bucket/prefix')
        s3_path = s3[len('s3://'):]
        parts = s3_path.split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''
        upload_directory_to_s3(out_dir, bucket, prefix)
        if args.s3_remove_local:
            import shutil
            shutil.rmtree(out_dir)
            logger.info('Removed local shards after upload')


if __name__ == '__main__':
    main()