"""Export taxonomy to CSV or JSON.

Usage:
    python manage.py export_taxonomy                    # writes to stdout (CSV)
    python manage.py export_taxonomy --out taxonomy.csv
    python manage.py export_taxonomy --out taxonomy.json --format json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from django.core.management.base import BaseCommand
from apps.catalog.models import Category, CategoryFacet


# CSV headers for export
HEADERS = [
    'node_id', 'name', 'slug', 'parent_slug', 'path', 'depth',
    'is_visible', 'is_deleted', 'meta_title', 'meta_description',
    'allowed_facets', 'product_count'
]


class Command(BaseCommand):
    help = 'Export categories to CSV or JSON.'

    def add_arguments(self, parser):
        parser.add_argument('--out', type=str, help='Output file path (CSV or JSON)')
        parser.add_argument(
            '--format',
            type=str,
            choices=['csv', 'json'],
            default='csv',
            help='Output format: csv or json (default: csv)'
        )
        parser.add_argument(
            '--include-deleted',
            action='store_true',
            help='Include soft-deleted categories'
        )

    def handle(self, *args, **options):
        out = options.get('out')
        output_format = options.get('format', 'csv')
        include_deleted = options.get('include_deleted', False)

        # Infer format from file extension if provided
        if out:
            if out.endswith('.json'):
                output_format = 'json'
            elif out.endswith('.csv'):
                output_format = 'csv'

        # Query categories
        qs = Category.objects.all().order_by('path')
        if not include_deleted:
            qs = qs.filter(is_deleted=False)

        # Prefetch facets
        qs = qs.prefetch_related('category_facets__facet')

        rows = []
        for c in qs:
            parent_slug = c.parent.slug if c.parent else ''
            # Get allowed facets for this category
            allowed_facets = ','.join([
                cf.facet.slug for cf in c.category_facets.all()
            ])

            rows.append({
                'node_id': str(c.id),
                'name': c.name,
                'slug': c.slug,
                'parent_slug': parent_slug,
                'path': c.path,
                'depth': c.depth,
                'is_visible': str(c.is_visible),
                'is_deleted': str(c.is_deleted),
                'meta_title': c.meta_title or '',
                'meta_description': c.meta_description or '',
                'allowed_facets': allowed_facets,
                'product_count': c.product_count,
            })

        if output_format == 'json':
            self._export_json(rows, out)
        else:
            self._export_csv(rows, out)

    def _export_csv(self, rows: List[Dict[str, Any]], out: str | None):
        """Export to CSV format."""
        if out:
            p = Path(out)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=HEADERS)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            self.stdout.write(self.style.SUCCESS(f'Exported {len(rows)} categories to {p}'))
        else:
            writer = csv.DictWriter(self.stdout, fieldnames=HEADERS)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def _export_json(self, rows: List[Dict[str, Any]], out: str | None):
        """Export to JSON format (nested tree structure)."""
        # Build nested tree
        tree = self._build_tree(rows)
        
        output = {
            'version': '1.0',
            'categories': tree,
            'total_count': len(rows)
        }
        
        if out:
            p = Path(out)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('w', encoding='utf-8') as fh:
                json.dump(output, fh, indent=2, ensure_ascii=False)
            self.stdout.write(self.style.SUCCESS(f'Exported {len(rows)} categories to {p}'))
        else:
            self.stdout.write(json.dumps(output, indent=2, ensure_ascii=False))

    def _build_tree(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build a nested tree structure from flat rows."""
        # Create lookup by slug
        by_slug = {r['slug']: r for r in rows}
        
        # Add children list to each node
        for r in rows:
            r['children'] = []
        
        # Build tree
        roots = []
        for r in rows:
            parent_slug = r.get('parent_slug')
            if parent_slug and parent_slug in by_slug:
                by_slug[parent_slug]['children'].append(r)
            else:
                roots.append(r)
        
        # Clean up: remove empty children lists and internal fields
        def clean_node(node):
            cleaned = {
                'name': node['name'],
                'slug': node['slug'],
            }
            if node.get('allowed_facets'):
                cleaned['facets'] = node['allowed_facets'].split(',')
            if node['children']:
                cleaned['children'] = [clean_node(c) for c in node['children']]
            return cleaned
        
        return [clean_node(r) for r in roots]
