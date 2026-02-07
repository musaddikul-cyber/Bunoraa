import random
import uuid
from django.core.management.base import BaseCommand
from django.utils import timezone
from decimal import Decimal
from apps.referral.models import ReferralCode, ReferralReward
from django.contrib.auth import get_user_model

User = get_user_model()


class Command(BaseCommand):
    help = 'Seeds the database with sample Referral data'

    def handle(self, *args, **options):
        self.stdout.write('Seeding Referral data...')

        users = list(User.objects.filter(is_superuser=False))

        if not users:
            self.stdout.write('  No users available to create referral data.')
            return

        created_codes_count = 0
        created_rewards_count = 0

        for user in random.sample(users, min(5, len(users))): # Create codes for a few random users
            # Ensure the user has a referral code
            code, created = ReferralCode.objects.get_or_create(user=user)
            if created:
                created_codes_count += 1
                self.stdout.write(f"  Created referral code '{code.code}' for user {user.email}")
            
            # Create some rewards for these referrers
            if random.random() > 0.5: # 50% chance to create a reward
                referee_user = random.choice([u for u in users if u != user]) # A different user
                if not referee_user: continue

                reward, created_reward = ReferralReward.objects.get_or_create(
                    referral_code=code,
                    referrer_user=user,
                    referee_user=referee_user,
                    defaults={
                        'reward_type': random.choice(['discount', 'store_credit']),
                        'value': Decimal(random.randint(5, 20)),
                        'description': f"{random.randint(5, 20)}% off or ${random.randint(5, 20)} store credit",
                        'status': random.choice(['pending', 'earned', 'applied']),
                        'earned_at': timezone.now() if random.random() > 0.5 else None,
                        'applied_at': timezone.now() if random.random() > 0.5 else None,
                    }
                )
                if created_reward:
                    created_rewards_count += 1
                    self.stdout.write(f"  Created reward for {user.email} (referee: {referee_user.email})")

        self.stdout.write(self.style.SUCCESS(f'Successfully created {created_codes_count} referral codes and {created_rewards_count} rewards.'))