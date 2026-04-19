# Generated migration for comprehensive banner management
from django.db import migrations, models
from colorfield.fields import ColorField


class Migration(migrations.Migration):
    dependencies = [
        ('promotions', '0010_banner_button_alignment_banner_button_font_size_and_more'),
    ]

    operations = [
        # Text colors
        migrations.AddField(
            model_name='banner',
            name='title_color',
            field=ColorField(blank=True, null=True, verbose_name='title color'),
        ),
        migrations.AddField(
            model_name='banner',
            name='subtitle_color',
            field=ColorField(blank=True, null=True, verbose_name='subtitle color'),
        ),
        
        # Font families
        migrations.AddField(
            model_name='banner',
            name='title_font_family',
            field=models.CharField(
                max_length=100, blank=True,
                choices=[
                    ('', 'Default'),
                    ('system-ui, -apple-system, sans-serif', 'System'),
                    ('Georgia, serif', 'Georgia (Serif)'),
                    ('Times New Roman, serif', 'Times (Serif)'),
                    ('Helvetica, Arial, sans-serif', 'Helvetica (Sans)'),
                    ('Roboto, sans-serif', 'Roboto'),
                    ('Open Sans, sans-serif', 'Open Sans'),
                    ('Lato, sans-serif', 'Lato'),
                    ('Montserrat, sans-serif', 'Montserrat'),
                    ('Poppins, sans-serif', 'Poppins'),
                    ('Playfair Display, serif', 'Playfair (Elegant)'),
                ],
                default='',
                verbose_name='title font family'
            ),
        ),
        migrations.AddField(
            model_name='banner',
            name='subtitle_font_family',
            field=models.CharField(
                max_length=100, blank=True,
                choices=[
                    ('', 'Default'),
                    ('system-ui, -apple-system, sans-serif', 'System'),
                    ('Georgia, serif', 'Georgia (Serif)'),
                    ('Times New Roman, serif', 'Times (Serif)'),
                    ('Helvetica, Arial, sans-serif', 'Helvetica (Sans)'),
                    ('Roboto, sans-serif', 'Roboto'),
                    ('Open Sans, sans-serif', 'Open Sans'),
                    ('Lato, sans-serif', 'Lato'),
                    ('Montserrat, sans-serif', 'Montserrat'),
                    ('Poppins, sans-serif', 'Poppins'),
                ],
                default='',
                verbose_name='subtitle font family'
            ),
        ),
        
        # Button colors
        migrations.AddField(
            model_name='banner',
            name='button_background_color',
            field=ColorField(blank=True, null=True, verbose_name='button background color'),
        ),
        migrations.AddField(
            model_name='banner',
            name='button_text_color',
            field=ColorField(blank=True, null=True, verbose_name='button text color'),
        ),
        migrations.AddField(
            model_name='banner',
            name='button_hover_background_color',
            field=ColorField(blank=True, null=True, verbose_name='button hover background color'),
        ),
        migrations.AddField(
            model_name='banner',
            name='button_hover_text_color',
            field=ColorField(blank=True, null=True, verbose_name='button hover text color'),
        ),
        
        # Animations and transitions
        migrations.AddField(
            model_name='banner',
            name='animation_type',
            field=models.CharField(
                max_length=50, blank=True,
                choices=[
                    ('', 'None'),
                    ('fade', 'Fade In'),
                    ('slide-up', 'Slide Up'),
                    ('slide-down', 'Slide Down'),
                    ('slide-left', 'Slide Left'),
                    ('slide-right', 'Slide Right'),
                    ('zoom', 'Zoom In'),
                    ('bounce', 'Bounce'),
                    ('flip', 'Flip'),
                ],
                default='fade',
                verbose_name='animation type'
            ),
        ),
        migrations.AddField(
            model_name='banner',
            name='transition_duration',
            field=models.DecimalField(
                max_digits=4, decimal_places=2, blank=True, null=True,
                verbose_name='transition duration (seconds)',
                help_text='Duration of the transition animation (e.g., 0.5)'
            ),
        ),
        
        # Banner timing for carousel
        migrations.AddField(
            model_name='banner',
            name='autoplay_delay',
            field=models.PositiveIntegerField(
                blank=True, null=True,
                verbose_name='autoplay delay (seconds)',
                help_text='Time before auto-rotating to next banner (leave empty to use default)'
            ),
        ),
        
        # Banner size presets (width management)
        migrations.AddField(
            model_name='banner',
            name='size_preset',
            field=models.CharField(
                max_length=30, blank=True,
                choices=[
                    ('', 'Default'),
                    ('compact', 'Compact (280px)'),
                    ('small', 'Small (350px)'),
                    ('medium', 'Medium (420px)'),
                    ('large', 'Large (520px)'),
                    ('hero', 'Hero (600px)'),
                    ('fullscreen', 'Fullscreen (100vh)'),
                    ('custom', 'Custom (use height field)'),
                ],
                default='medium',
                verbose_name='size preset'
            ),
        ),
        
        # Opacity/Transparency
        migrations.AddField(
            model_name='banner',
            name='container_opacity',
            field=models.DecimalField(
                max_digits=3, decimal_places=2, blank=True, null=True,
                verbose_name='container opacity',
                help_text='Overall banner opacity from 0 (transparent) to 1 (opaque)'
            ),
        ),
        
        # Background image settings
        migrations.AddField(
            model_name='banner',
            name='background_size',
            field=models.CharField(
                max_length=30, blank=True,
                choices=[
                    ('cover', 'Cover (default)'),
                    ('contain', 'Contain'),
                    ('auto', 'Auto'),
                    ('100% 100%', 'Stretch'),
                ],
                default='cover',
                verbose_name='background size'
            ),
        ),
        migrations.AddField(
            model_name='banner',
            name='background_position',
            field=models.CharField(
                max_length=30, blank=True,
                choices=[
                    ('center', 'Center'),
                    ('top', 'Top'),
                    ('bottom', 'Bottom'),
                    ('left', 'Left'),
                    ('right', 'Right'),
                    ('top left', 'Top Left'),
                    ('top right', 'Top Right'),
                    ('bottom left', 'Bottom Left'),
                    ('bottom right', 'Bottom Right'),
                ],
                default='center',
                verbose_name='background position'
            ),
        ),
        
        # Mobile-specific settings
        migrations.AddField(
            model_name='banner',
            name='mobile_height',
            field=models.CharField(
                max_length=20, blank=True,
                verbose_name='mobile height',
                help_text='CSS height for mobile devices (e.g., 280px, 50vh)'
            ),
        ),
        migrations.AddField(
            model_name='banner',
            name='hide_on_mobile',
            field=models.BooleanField(
                default=False,
                verbose_name='hide on mobile'
            ),
        ),
        migrations.AddField(
            model_name='banner',
            name='hide_on_desktop',
            field=models.BooleanField(
                default=False,
                verbose_name='hide on desktop'
            ),
        ),
    ]
