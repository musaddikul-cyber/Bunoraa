"""Management command to seed categories with a robust hierarchical tree.

- Idempotent: creates or updates categories by (parent, slug).
- Optional `--force` flag will soft-delete existing categories before seeding.
- Optional `--assign-facets` flag assigns default facets to top-level categories.
- Optional `--file` allows loading taxonomy from a JSON file.

Usage:
    python manage.py seed_categories
    python manage.py seed_categories --force
    python manage.py seed_categories --assign-facets
    python manage.py seed_categories --file=taxonomy.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Union

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.text import slugify

from apps.catalog.models import Category, Facet, CategoryFacet


# Default taxonomy file location
DEFAULT_TAXONOMY_PATH = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'taxonomy.json'


# ============================================================================
# CATEGORY TREE - Bunoraa Artisan Marketplace Categories
# ============================================================================
CATEGORY_TREE: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------
    # 1. Home & Living
    # ------------------------------------------------------------------
    {
        'code': 'CAT_HOME',
        'name': 'Home & Living',
        'slug': 'home-living',
        'children': [
            {
                'code': 'CAT_HOME_DECOR',
                'name': 'Decor',
                'slug': 'decor',
                'children': [
                    {'code': 'CAT_HOME_DECOR_WALL', 'name': 'Wall Art', 'slug': 'wall-art'},
                    {'code': 'CAT_HOME_DECOR_SCULPT', 'name': 'Sculptures', 'slug': 'sculptures'},
                    {'code': 'CAT_HOME_DECOR_CANDLE', 'name': 'Candle Holders', 'slug': 'candle-holders'},
                    {'code': 'CAT_HOME_DECOR_PLANT', 'name': 'Planters', 'slug': 'planters'},
                    {'code': 'CAT_HOME_DECOR_LIGHT', 'name': 'Lighting', 'slug': 'lighting'},
                ]
            },
            {
                'code': 'CAT_HOME_TEXTILES',
                'name': 'Textiles',
                'slug': 'textiles',
                'children': [
                    {'code': 'CAT_HOME_TEXT_RUGS', 'name': 'Rugs & Mats', 'slug': 'rugs-mats'},
                    {'code': 'CAT_HOME_TEXT_CUSH', 'name': 'Cushion Covers', 'slug': 'cushion-covers'},
                    {'code': 'CAT_HOME_TEXT_THROWS', 'name': 'Throws & Blankets', 'slug': 'throws-blankets'},
                    {'code': 'CAT_HOME_TEXT_RUNNER', 'name': 'Table Runners', 'slug': 'table-runners'},
                    {'code': 'CAT_HOME_TEXT_BED', 'name': 'Bedding', 'slug': 'bedding'},
                    {'code': 'CAT_HOME_TEXT_CURTAIN', 'name': 'Curtains', 'slug': 'curtains'},
                ]
            },
            {
                'code': 'CAT_HOME_KITCHEN',
                'name': 'Kitchen',
                'slug': 'kitchen',
                'children': [
                    {'code': 'CAT_HOME_KITCH_CERAMIC', 'name': 'Ceramics', 'slug': 'ceramics'},
                    {'code': 'CAT_HOME_KITCH_WOOD', 'name': 'Wooden Ware', 'slug': 'wooden-ware'},
                    {'code': 'CAT_HOME_KITCH_UTENSIL', 'name': 'Utensils', 'slug': 'utensils'},
                    {'code': 'CAT_HOME_KITCH_TRAY', 'name': 'Serving Trays', 'slug': 'serving-trays'},
                    {'code': 'CAT_HOME_KITCH_DRINK', 'name': 'Drinkware', 'slug': 'drinkware'},
                ]
            },
            {
                'code': 'CAT_HOME_FURN',
                'name': 'Furniture',
                'slug': 'furniture',
                'children': [
                    {'code': 'CAT_HOME_FURN_STOOL', 'name': 'Stools & Benches', 'slug': 'stools-benches'},
                    {'code': 'CAT_HOME_FURN_TABLE', 'name': 'Accent Tables', 'slug': 'accent-tables'},
                    {'code': 'CAT_HOME_FURN_CHAIR', 'name': 'Wooden Chairs', 'slug': 'wooden-chairs'},
                    {'code': 'CAT_HOME_FURN_STORE', 'name': 'Storage', 'slug': 'storage'},
                ]
            },
            {
                'code': 'CAT_HOME_OUTDOOR',
                'name': 'Outdoor',
                'slug': 'outdoor',
                'children': [
                    {'code': 'CAT_HOME_OUT_PLANT', 'name': 'Planters & Pots', 'slug': 'planters-pots'},
                ]
            },
        ]
    },

    # ------------------------------------------------------------------
    # 2. Fashion & Apparel
    # ------------------------------------------------------------------
    {
        'code': 'CAT_APPAREL',
        'name': 'Fashion & Apparel',
        'slug': 'fashion-apparel',
        'children': [
            {
                'code': 'CAT_APP_WOMEN',
                'name': 'Women',
                'slug': 'women',
                'children': [
                    {'code': 'CAT_APP_W_DRESS', 'name': 'Dresses', 'slug': 'dresses'},
                    {'code': 'CAT_APP_W_TUNIC', 'name': 'Tunics & Tops', 'slug': 'tunics-tops'},
                    {'code': 'CAT_APP_W_SAREE', 'name': 'Sarees', 'slug': 'sarees'},
                    {'code': 'CAT_APP_W_SHAWL', 'name': 'Shawls', 'slug': 'shawls'},
                ]
            },
            {
                'code': 'CAT_APP_MEN',
                'name': 'Men',
                'slug': 'men',
                'children': [
                    {'code': 'CAT_APP_M_KURTA', 'name': 'Kurtas', 'slug': 'kurtas'},
                    {'code': 'CAT_APP_M_SHIRT', 'name': 'Shirts', 'slug': 'shirts'},
                ]
            },
            {
                'code': 'CAT_APP_KIDS',
                'name': 'Kids',
                'slug': 'kids',
            },
            {
                'code': 'CAT_APP_BAG',
                'name': 'Bags',
                'slug': 'bags',
                'children': [
                    {'code': 'CAT_APP_BAG_TOTE', 'name': 'Totes', 'slug': 'totes'},
                    {'code': 'CAT_APP_BAG_CLUTCH', 'name': 'Clutches', 'slug': 'clutches'},
                    {'code': 'CAT_APP_BAG_BACK', 'name': 'Backpacks', 'slug': 'backpacks'},
                    {'code': 'CAT_APP_BAG_LEATHER', 'name': 'Leather Bags', 'slug': 'leather-bags'},
                ]
            },
            {
                'code': 'CAT_APP_FOOT',
                'name': 'Footwear',
                'slug': 'footwear',
                'children': [
                    {'code': 'CAT_APP_FOOT_SANDAL', 'name': 'Sandals', 'slug': 'sandals'},
                    {'code': 'CAT_APP_FOOT_SLIP', 'name': 'Slippers', 'slug': 'slippers'},
                ]
            },
            {
                'code': 'CAT_APP_ACC',
                'name': 'Accessories',
                'slug': 'accessories',
                'children': [
                    {'code': 'CAT_APP_ACC_BELT', 'name': 'Belts', 'slug': 'belts'},
                    {'code': 'CAT_APP_ACC_HAIR', 'name': 'Hair Accessories', 'slug': 'hair-accessories'},
                    {'code': 'CAT_APP_ACC_SCARF', 'name': 'Scarves', 'slug': 'scarves'},
                ]
            },
        ]
    },

    # ------------------------------------------------------------------
    # 3. Jewelry
    # ------------------------------------------------------------------
    {
        'code': 'CAT_JEWELRY',
        'name': 'Jewelry',
        'slug': 'jewelry',
        'children': [
            {
                'code': 'CAT_JEW_NECK',
                'name': 'Necklaces',
                'slug': 'necklaces',
                'children': [
                    {'code': 'CAT_JEW_NECK_BEAD', 'name': 'Beaded', 'slug': 'beaded'},
                    {'code': 'CAT_JEW_NECK_SILVER', 'name': 'Silver', 'slug': 'silver'},
                    {'code': 'CAT_JEW_NECK_MIN', 'name': 'Minimalist', 'slug': 'minimalist'},
                    {'code': 'CAT_JEW_NECK_TRIBAL', 'name': 'Tribal', 'slug': 'tribal'},
                ]
            },
            {
                'code': 'CAT_JEW_EAR',
                'name': 'Earrings',
                'slug': 'earrings',
                'children': [
                    {'code': 'CAT_JEW_EAR_STUD', 'name': 'Studs', 'slug': 'studs'},
                    {'code': 'CAT_JEW_EAR_DANGLE', 'name': 'Danglers', 'slug': 'danglers'},
                    {'code': 'CAT_JEW_EAR_HOOP', 'name': 'Hoops', 'slug': 'hoops'},
                ]
            },
            {'code': 'CAT_JEW_BRACE', 'name': 'Bracelets', 'slug': 'bracelets'},
            {'code': 'CAT_JEW_RING', 'name': 'Rings', 'slug': 'rings'},
            {'code': 'CAT_JEW_ANKLET', 'name': 'Anklets', 'slug': 'anklets'},
            {'code': 'CAT_JEW_BODY', 'name': 'Body Jewelry', 'slug': 'body-jewelry'},
            {'code': 'CAT_JEW_SET', 'name': 'Sets', 'slug': 'sets'},
            {'code': 'CAT_JEW_MEMORIAL', 'name': 'Memorial', 'slug': 'memorial'},
        ]
    },

    # ------------------------------------------------------------------
    # 4. Art & Collectibles
    # ------------------------------------------------------------------
    {
        'code': 'CAT_ART',
        'name': 'Art & Collectibles',
        'slug': 'art-collectibles',
        'children': [
            {
                'code': 'CAT_ART_PAINT',
                'name': 'Paintings',
                'slug': 'paintings',
                'children': [
                    {'code': 'CAT_ART_PAINT_ACR', 'name': 'Acrylic', 'slug': 'acrylic'},
                    {'code': 'CAT_ART_PAINT_WATER', 'name': 'Watercolor', 'slug': 'watercolor'},
                    {'code': 'CAT_ART_PAINT_MIXED', 'name': 'Mixed Media', 'slug': 'mixed-media'},
                    {'code': 'CAT_ART_PAINT_FOLK', 'name': 'Folk Art', 'slug': 'folk-art'},
                    {'code': 'CAT_ART_PAINT_MINI', 'name': 'Miniature', 'slug': 'miniature'},
                ]
            },
            {'code': 'CAT_ART_PRINT', 'name': 'Prints', 'slug': 'prints'},
            {'code': 'CAT_ART_SCULPT', 'name': 'Sculptures', 'slug': 'sculptures'},
            {'code': 'CAT_ART_CRAFT', 'name': 'Crafts', 'slug': 'crafts'},
            {'code': 'CAT_ART_LTD', 'name': 'Limited Editions', 'slug': 'limited-editions'},
            {'code': 'CAT_ART_DOLL', 'name': 'Dolls', 'slug': 'dolls'},
            {'code': 'CAT_ART_FIBER', 'name': 'Fiber Art', 'slug': 'fiber-art'},
            {'code': 'CAT_ART_GLASS', 'name': 'Glass Art', 'slug': 'glass-art'},
        ]
    },

    # ------------------------------------------------------------------
    # 5. Gifts & Occasions
    # ------------------------------------------------------------------
    {
        'code': 'CAT_GIFTS',
        'name': 'Gifts & Occasions',
        'slug': 'gifts-occasions',
        'children': [
            {
                'code': 'CAT_GIFT_SET',
                'name': 'Gift Sets',
                'slug': 'gift-sets',
                'children': [
                    {'code': 'CAT_GIFT_SET_HOME', 'name': 'Home Boxes', 'slug': 'home-boxes'},
                    {'code': 'CAT_GIFT_SET_JEW', 'name': 'Jewelry Packs', 'slug': 'jewelry-packs'},
                    {'code': 'CAT_GIFT_SET_CARE', 'name': 'Care Kits', 'slug': 'care-kits'},
                ]
            },
            {
                'code': 'CAT_GIFT_OCC',
                'name': 'By Occasion',
                'slug': 'by-occasion',
                'children': [
                    {'code': 'CAT_GIFT_OCC_BIRTH', 'name': 'Birthday', 'slug': 'birthday'},
                    {'code': 'CAT_GIFT_OCC_WED', 'name': 'Wedding', 'slug': 'wedding'},
                    {'code': 'CAT_GIFT_OCC_ANNIV', 'name': 'Anniversary', 'slug': 'anniversary'},
                    {'code': 'CAT_GIFT_OCC_HOUSE', 'name': 'Housewarming', 'slug': 'housewarming'},
                    {'code': 'CAT_GIFT_OCC_BABY', 'name': 'Baby Shower', 'slug': 'baby-shower'},
                    {'code': 'CAT_GIFT_OCC_CORP', 'name': 'Corporate', 'slug': 'corporate'},
                ]
            },
            {
                'code': 'CAT_GIFT_DISC',
                'name': 'Discover',
                'slug': 'discover',
                'children': [
                    {'code': 'CAT_GIFT_DISC_HIM', 'name': 'For Him', 'slug': 'for-him'},
                    {'code': 'CAT_GIFT_DISC_HER', 'name': 'For Her', 'slug': 'for-her'},
                    {'code': 'CAT_GIFT_DISC_FRIEND', 'name': 'For Friends', 'slug': 'for-friends'},
                    {'code': 'CAT_GIFT_DISC_COUPLE', 'name': 'For Couples', 'slug': 'for-couples'},
                    {'code': 'CAT_GIFT_DISC_KIDS', 'name': 'For Kids', 'slug': 'for-kids'},
                    {'code': 'CAT_GIFT_DISC_PETS', 'name': 'For Pets', 'slug': 'for-pets'},
                ]
            },
            {'code': 'CAT_GIFT_PERSON', 'name': 'Personalized', 'slug': 'personalized'},
        ]
    },

    # ------------------------------------------------------------------
    # 6. Personal Care & Wellness
    # ------------------------------------------------------------------
    {
        'code': 'CAT_WELLNESS',
        'name': 'Personal Care & Wellness',
        'slug': 'personal-care-wellness',
        'children': [
            {'code': 'CAT_WELL_SOAP', 'name': 'Soaps', 'slug': 'soaps'},
            {'code': 'CAT_WELL_SKIN', 'name': 'Skincare', 'slug': 'skincare'},
            {'code': 'CAT_WELL_BATH', 'name': 'Bath & Body', 'slug': 'bath-body'},
            {'code': 'CAT_WELL_AROMA', 'name': 'Aromatherapy', 'slug': 'aromatherapy'},
            {'code': 'CAT_WELL_HERBAL', 'name': 'Herbal Products', 'slug': 'herbal-products'},
            {'code': 'CAT_WELL_HAIR', 'name': 'Hair Care', 'slug': 'hair-care'},
            {'code': 'CAT_WELL_MAKEUP', 'name': 'Makeup', 'slug': 'makeup'},
            {'code': 'CAT_WELL_FRAG', 'name': 'Fragrances', 'slug': 'fragrances'},
        ]
    },

    # ------------------------------------------------------------------
    # 7. Kids & Baby
    # ------------------------------------------------------------------
    {
        'code': 'CAT_KIDS',
        'name': 'Kids & Baby',
        'slug': 'kids-baby',
        'children': [
            {
                'code': 'CAT_KIDS_TOY',
                'name': 'Toys',
                'slug': 'toys',
                'children': [
                    {'code': 'CAT_KIDS_TOY_WOOD', 'name': 'Wooden', 'slug': 'wooden'},
                    {'code': 'CAT_KIDS_TOY_SOFT', 'name': 'Soft', 'slug': 'soft'},
                ]
            },
            {'code': 'CAT_KIDS_APPAREL', 'name': 'Apparel', 'slug': 'apparel'},
            {'code': 'CAT_KIDS_NURSERY', 'name': 'Nursery Decor', 'slug': 'nursery-decor'},
            {'code': 'CAT_KIDS_ACC', 'name': 'Accessories', 'slug': 'accessories'},
            {'code': 'CAT_KIDS_CARE', 'name': 'Baby Care', 'slug': 'baby-care'},
        ]
    },

    # ------------------------------------------------------------------
    # 8. Seasonal & Featured
    # ------------------------------------------------------------------
    {
        'code': 'CAT_SEASONAL',
        'name': 'Seasonal & Featured',
        'slug': 'seasonal-featured',
        'children': [
            {
                'code': 'CAT_SEASON_FEST',
                'name': 'Festives',
                'slug': 'festives',
                'children': [
                    {'code': 'CAT_SEASON_FEST_EID', 'name': 'Eid', 'slug': 'eid'},
                    {'code': 'CAT_SEASON_FEST_PUJA', 'name': 'Puja', 'slug': 'puja'},
                    {'code': 'CAT_SEASON_FEST_WED', 'name': 'Wedding', 'slug': 'wedding'},
                    {'code': 'CAT_SEASON_FEST_HOL', 'name': 'Holiday', 'slug': 'holiday'},
                ]
            },
            {'code': 'CAT_SEASON_HOME', 'name': 'Home Picks', 'slug': 'home-picks'},
            {'code': 'CAT_SEASON_LTD', 'name': 'Limited', 'slug': 'limited'},
            {'code': 'CAT_SEASON_ARTIST', 'name': 'Artists', 'slug': 'artists'},
            {'code': 'CAT_SEASON_ESS', 'name': 'Essentials', 'slug': 'essentials'},
        ]
    },

    # ------------------------------------------------------------------
    # 9. Custom & Made-to-Order
    # ------------------------------------------------------------------
    {
        'code': 'CAT_CUSTOM',
        'name': 'Custom & Made-to-Order',
        'slug': 'custom-made-to-order',
        'children': [
            {'code': 'CAT_CUSTOM_JEW', 'name': 'Jewelry', 'slug': 'jewelry'},
            {'code': 'CAT_CUSTOM_DECOR', 'name': 'Decor', 'slug': 'decor'},
            {'code': 'CAT_CUSTOM_APP', 'name': 'Apparel', 'slug': 'apparel'},
            {'code': 'CAT_CUSTOM_GIFT', 'name': 'Gifts', 'slug': 'gifts'},
        ]
    },
]


# ============================================================================
# DEFAULT FACET MAPPING
# Assigns facets to top-level category codes for filtering capabilities.
# ============================================================================
DEFAULT_FACET_MAP: Dict[str, List[str]] = {
    'CAT_JEWELRY': ['material', 'color', 'style', 'occasion', 'gemstone'],
    'CAT_HOME': ['material', 'color', 'style', 'room', 'size'],
    'CAT_APPAREL': ['material', 'color', 'size', 'style', 'occasion', 'gender'],
    'CAT_WELLNESS': ['ingredient', 'skin_type', 'scent', 'organic'],
    'CAT_ART': ['medium', 'style', 'subject', 'size', 'frame_type'],
    'CAT_GIFTS': ['occasion', 'recipient', 'price_range', 'style'],
    'CAT_KIDS': ['age_range', 'material', 'educational', 'gender'],
    'CAT_SEASONAL': ['season', 'occasion', 'style'],
    'CAT_CUSTOM': ['customization_type', 'lead_time', 'material'],
}


class Command(BaseCommand):
    help = 'Seed the categories tree. Safe to run multiple times (idempotent).'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Soft-delete existing categories before seeding'
        )
        parser.add_argument(
            '--file',
            type=str,
            default=None,
            help='Path to taxonomy JSON file (optional)'
        )
        parser.add_argument(
            '--assign-facets',
            action='store_true',
            default=True,
            help='Assign default facets to top-level categories'
        )
        parser.add_argument(
            '--no-assign-facets',
            action='store_false',
            dest='assign_facets',
            help='Do not assign default facets'
        )

    def handle(self, *args, **options):
        force = options.get('force')
        taxonomy_file = options.get('file')
        assign_facets = options.get('assign_facets', True)

        if force:
            self.stdout.write(self.style.WARNING('Soft-deleting existing categories (force).'))
            from django.utils import timezone
            Category.objects.filter(is_deleted=False).update(
                is_deleted=True,
                deleted_at=timezone.now()
            )

        # Load taxonomy from file if provided
        if taxonomy_file and Path(taxonomy_file).exists():
            try:
                with open(taxonomy_file, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                    tree = data.get('categories', CATEGORY_TREE)
                self.stdout.write(f'Loaded taxonomy from: {taxonomy_file}')
            except Exception as exc:
                self.stderr.write(self.style.ERROR(f'Failed to load taxonomy file: {exc}'))
                tree = CATEGORY_TREE
        else:
            tree = CATEGORY_TREE

        with transaction.atomic():
            created = 0
            updated = 0
            # Track top-level categories by code for facet assignment
            top_level_cats: Dict[str, Category] = {}

            def create_node(node: Union[str, Dict[str, Any]], parent: Category | None = None, order: int = 0) -> Category | None:
                nonlocal created, updated

                if isinstance(node, str):
                    name = node
                    children = None
                    code = None
                    url_slug = slugify(name)
                else:
                    name = node.get('name') or node.get('display_name')
                    children = node.get('children')
                    code = node.get('code')
                    url_slug = node.get('slug') or node.get('url_slug') or slugify(name)

                defaults = {
                    'name': name,
                    'is_visible': True,
                    'is_deleted': False,
                }

                category, cat_created = Category.objects.get_or_create(
                    parent=parent,
                    slug=url_slug,
                    defaults=defaults
                )

                if cat_created:
                    created += 1
                    full_path = self._get_full_path(category)
                    self.stdout.write(self.style.SUCCESS(f'Created category: {full_path}'))
                else:
                    # Update name/is_visible if changed
                    changed = False
                    if category.name != name:
                        category.name = name
                        changed = True
                    if not category.is_visible:
                        category.is_visible = True
                        changed = True
                    if category.is_deleted:
                        category.is_deleted = False
                        category.deleted_at = None
                        changed = True
                    if changed:
                        category.save()
                        updated += 1
                        full_path = self._get_full_path(category)
                        self.stdout.write(self.style.NOTICE(f'Updated category: {full_path}'))

                # Track top-level categories by code for facet assignment
                if parent is None and code:
                    top_level_cats[code] = category

                # Recursively create children
                if children:
                    for idx, child in enumerate(children, start=1):
                        create_node(child, parent=category, order=idx)

                return category

            # Create all categories from the tree
            for idx, item in enumerate(tree, start=1):
                create_node(item, parent=None, order=idx)

            # Assign facets to top-level categories if requested
            facets_assigned = 0
            if assign_facets:
                facets_assigned = self._assign_facets(top_level_cats)

            self.stdout.write(self.style.SUCCESS(
                f'Seeding complete: {created} created, {updated} updated, {facets_assigned} facet assignments.'
            ))

    def _get_full_path(self, category):
        """Build full path string for a category."""
        parts = [category.name]
        parent = category.parent
        while parent:
            parts.insert(0, parent.name)
            parent = parent.parent
        return ' > '.join(parts)

    def _assign_facets(self, top_level_cats: Dict[str, 'Category']) -> int:
        """Assign default facets to top-level categories based on DEFAULT_FACET_MAP."""
        assigned = 0

        for cat_code, facet_codes in DEFAULT_FACET_MAP.items():
            category = top_level_cats.get(cat_code)
            if not category:
                continue

            for facet_code in facet_codes:
                # Get or create the facet
                facet, facet_created = Facet.objects.get_or_create(
                    slug=facet_code,
                    defaults={
                        'name': facet_code.replace('_', ' ').title(),
                        'type': 'choice',
                    }
                )

                if facet_created:
                    self.stdout.write(f'  Created facet: {facet.name}')

                # Create the category-facet relationship
                _, cf_created = CategoryFacet.objects.get_or_create(
                    category=category,
                    facet=facet
                )

                if cf_created:
                    assigned += 1
                    self.stdout.write(f'  Assigned facet "{facet.name}" to {category.name}')

        return assigned
