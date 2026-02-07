import random
from django.core.management.base import BaseCommand
from django.utils import timezone
from apps.chat.models import Conversation, Message, ConversationStatus, ChatAgent
from django.contrib.auth import get_user_model

User = get_user_model()


class Command(BaseCommand):
    help = 'Seeds the database with sample Chat data'

    def handle(self, *args, **options):
        self.stdout.write('Seeding Chat data...')

        users = list(User.objects.filter(is_superuser=False))
        admin_user = User.objects.filter(is_superuser=True).first()
        agents = list(ChatAgent.objects.filter(is_active=True, is_online=True, is_accepting_chats=True))

        if not users:
            self.stdout.write('  No regular users available to create conversations.')
            return
        if not admin_user:
            self.stdout.write('  No admin user found.')
            return

        # Create an open conversation for a regular user with the admin as agent
        if agents:
            customer_user = random.choice(users)
            agent = random.choice(agents)

            conversation, created = Conversation.objects.get_or_create(
                customer=customer_user,
                agent=agent,
                status=ConversationStatus.ACTIVE,
                defaults={
                    'topic': 'Product Inquiry',
                    'last_message_at': timezone.now(),
                }
            )
            if created:
                self.stdout.write(f"  Created active conversation for {customer_user.email} with agent {agent.user.email}.")
                Message.objects.create(
                    conversation=conversation,
                    sender=customer_user,
                    content="Hello, I have a question about one of your products.",
                    is_from_customer=True,
                )
                Message.objects.create(
                    conversation=conversation,
                    sender=agent.user,
                    content="Hi! How can I help you today?",
                    is_from_customer=False,
                )
        else:
            self.stdout.write('  No active chat agents found to assign to conversation.')

        # Create a resolved conversation for a regular user
        customer_user = random.choice(users)
        conversation_resolved, created_resolved = Conversation.objects.get_or_create(
            customer=customer_user,
            status=ConversationStatus.RESOLVED,
            defaults={
                'topic': 'Order Issue',
                'last_message_at': timezone.now() - timezone.timedelta(days=1),
                'resolved_at': timezone.now(),
            }
        )
        if created_resolved:
            self.stdout.write(f"  Created resolved conversation for {customer_user.email}.")

        self.stdout.write(self.style.SUCCESS('Chat data seeding completed.'))