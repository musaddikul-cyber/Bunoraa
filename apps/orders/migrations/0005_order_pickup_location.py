from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('contacts', '0002_customizationrequest'),
        ('orders', '0004_order_currency_order_exchange_rate'),
    ]

    operations = [
        migrations.AddField(
            model_name='order',
            name='pickup_location',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='order_pickups',
                to='contacts.storelocation',
            ),
        ),
        migrations.AlterField(
            model_name='order',
            name='shipping_method',
            field=models.CharField(
                choices=[
                    ('standard', 'Standard Shipping'),
                    ('express', 'Express Shipping'),
                    ('overnight', 'Overnight Shipping'),
                    ('pickup', 'Store Pickup'),
                ],
                db_index=True,
                default='standard',
                help_text='Shipping method chosen for this order',
                max_length=20,
            ),
        ),
    ]
