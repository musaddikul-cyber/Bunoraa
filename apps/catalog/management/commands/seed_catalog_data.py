import random
from django.core.management.base import BaseCommand
from django.utils import timezone
from apps.catalog.models import Product, ProductMakingOf, ProductQuestion, ProductAnswer
from django.contrib.auth import get_user_model

User = get_user_model()


class Command(BaseCommand):
    help = 'Seeds the database with sample Catalog app data (MakingOf, Q&A)'

    def handle(self, *args, **options):
        self.stdout.write('Seeding Catalog app data...')

        products = list(Product.objects.all())
        users = list(User.objects.filter(is_superuser=False))

        if not products:
            self.stdout.write('  No products available to seed catalog data.')
            return
        if not users:
            self.stdout.write('  No users available to seed Q&A data.')
        
        # Seed ProductMakingOf
        created_making_of_count = 0
        for product in random.sample(products, min(5, len(products))): # Add steps to a few random products
            num_steps = random.randint(2, 4)
            for i in range(num_steps):
                step, created = ProductMakingOf.objects.get_or_create(
                    product=product,
                    order=i + 1,
                    defaults={
                        'title': f'Step {i+1}: {product.name} - {random.choice(["Design", "Embroidery", "Finishing", "Packaging"])}',
                        'description': f'Details about the {random.choice(["design", "crafting", "final touches"])} process for this {product.name.lower()}: {random.choice(["careful selection of materials", "intricate hand-stitching techniques", "quality checks and sustainable packaging"])}.',
                        'image': f'catalog/making_of/default_step_{random.randint(1, 3)}.jpg', # Placeholder image
                        'video_url': random.choice(['https://www.youtube.com/watch?v=dQw4w9WgXcQ', '']), # Placeholder video
                    }
                )
                if created:
                    created_making_of_count += 1
        self.stdout.write(f'  Created {created_making_of_count} making-of steps.')

        # Seed ProductQuestion and ProductAnswer
        created_q_count = 0
        created_a_count = 0
        for product in random.sample(products, min(10, len(products))): # Add Q&A to a few random products
            if not users: continue # Skip if no users to ask questions
            
            question_text = f"Is the {product.name} available in {random.choice(['red', 'blue', 'green'])}?"
            question, created = ProductQuestion.objects.get_or_create(
                product=product,
                user=random.choice(users),
                question_text=question_text,
                defaults={
                    'status': random.choice(['pending', 'approved']),
                }
            )
            if created:
                created_q_count += 1
                self.stdout.write(f"  Created question for {product.name}.")

                if question.status == 'approved' and users: # Only answer approved questions
                    answer_text = f"Yes, the {product.name} is available in {random.choice(['red', 'blue', 'green'])}? Please check variants."
                    answer, created_a = ProductAnswer.objects.get_or_create(
                        question=question,
                        user=random.choice(users), # Could be admin user
                        answer_text=answer_text,
                        defaults={
                            'status': 'approved',
                        }
                    )
                    if created_a:
                        created_a_count += 1
                        self.stdout.write(f"  Created answer for question '{question.question_text[:20]}'.")

        self.stdout.write(f'Successfully created {created_q_count} questions and {created_a_count} answers.')
        self.stdout.write(self.style.SUCCESS('Catalog app data seeding completed.'))