"""
Build Similarity Index for Product Images

This management command generates visual feature embeddings for all product images
and builds a similarity index for use in visual recommendation models.
"""
import logging
from PIL import Image
import numpy as np

from django.core.management.base import BaseCommand
from django.core.files.storage import default_storage

from apps.catalog.models import Product
from ml.models.vision import ProductImageClassifier

logger = logging.getLogger("bunoraa.ml")


class Command(BaseCommand):
    help = "Builds the visual similarity index for product images."

    def handle(self, *args, **options):
        self.stdout.write("Starting to build visual similarity index...")

        try:
            import requests
            from io import BytesIO
            from django.conf import settings
            from ml.core.registry import get_registry, ModelStatus # Import these

            # Initialize the vision model
            model = ProductImageClassifier(model_name="visual_similarity_classifier", version="1.0.0")
            if not model.model:
                model.model = model.build_model()

            # Get all active products that have at least one image
            products = Product.objects.filter(is_active=True, images__isnull=False).distinct()
            
            product_ids = []
            images = []
            
            self.stdout.write(f"Processing {products.count()} products...")

            media_url_base = settings.MEDIA_URL # Use settings.MEDIA_URL

            # Ensure media_url_base ends with a slash and is an absolute URL
            if not media_url_base.endswith('/'):
                media_url_base += '/'
            if not media_url_base.startswith('http'): # Assuming external storage, construct full URL
                self.stderr.write(self.style.WARNING(
                    "settings.MEDIA_URL is not an absolute URL. "
                    "Assuming it needs to be prefixed with the site's base URL or "
                    "that an absolute MEDIA_URL is configured in environment variables."
                ))

            for product in products:
                # Get the primary image for the product
                primary_image = product.images.filter(is_primary=True).first()
                if not primary_image:
                    # If no primary, take the first one
                    primary_image = product.images.first()

                if primary_image and primary_image.image:
                    try:
                        # Construct the full image URL using settings.MEDIA_URL
                        image_url = f"{media_url_base}{primary_image.image.name}"
                        response = requests.get(image_url, stream=True)
                        response.raise_for_status() # Raise an exception for bad status codes
                        
                        img = Image.open(BytesIO(response.content)).convert('RGB')
                        images.append(np.array(img))
                        product_ids.append(product.id)
                    except requests.exceptions.RequestException as e:
                        self.stderr.write(f"Could not download image for product {product.id} from {image_url}: {e}")
                    except Exception as e:
                        self.stderr.write(f"Could not process image for product {product.id}: {e}")
            
            if not images:
                self.stderr.write("No images found to build the index. Aborting.")
                return

            # Build the similarity index
            model.build_similarity_index(product_ids, images)

            # Register the model with the registry
            registry = get_registry()
            registered_entry = registry.register(
                model=model,
                model_name="visual_similarity_classifier",
                version="1.0.0", # Use a proper versioning scheme in real projects
                model_type="vision",
                framework="pytorch",
                description="Product image classifier with visual similarity index for product recommendations.",
                tags=["visual_search", "recommendation"]
            )
            
            self.stdout.write(self.style.SUCCESS(f"Successfully built visual similarity index and registered model: {registered_entry.model_id}"))

        except Exception as e:
            self.stderr.write(self.style.ERROR(f"An error occurred: {e}"))
