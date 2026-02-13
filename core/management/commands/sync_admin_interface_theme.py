from django.core.management.base import BaseCommand

from admin_interface.models import Theme


class Command(BaseCommand):
    help = "Sync Bunoraa admin-interface themes with frontend design tokens."

    def handle(self, *args, **options):
        light_theme = {
            "name": "Bunoraa Light",
            "title": "Bunoraa Administration",
            "title_visible": True,
            "logo_visible": True,
            "logo_color": "#8b3b18",
            "title_color": "#8b3b18",
            "env_color": "#f7a708",
            "env_visible_in_header": True,
            "env_visible_in_favicon": True,
            "language_chooser_active": True,
            "language_chooser_control": "minimal-select",
            "language_chooser_display": "name",
            "css_header_background_color": "#ffffff",
            "css_header_text_color": "#201e1d",
            "css_header_link_color": "#8b3b18",
            "css_header_link_hover_color": "#af3625",
            "css_module_background_color": "#8b3b18",
            "css_module_background_selected_color": "#fce7e4",
            "css_module_text_color": "#ffffff",
            "css_module_link_color": "#ffffff",
            "css_module_link_selected_color": "#ffffff",
            "css_module_link_hover_color": "#fef3c7",
            "css_module_rounded_corners": True,
            "css_generic_link_color": "#8b3b18",
            "css_generic_link_hover_color": "#af3625",
            "css_generic_link_active_color": "#d14430",
            "css_save_button_background_color": "#8b3b18",
            "css_save_button_background_hover_color": "#793315",
            "css_save_button_text_color": "#ffffff",
            "css_delete_button_background_color": "#ef4444",
            "css_delete_button_background_hover_color": "#dc2626",
            "css_delete_button_text_color": "#ffffff",
            "related_modal_active": True,
            "related_modal_background_color": "#201e1d",
            "related_modal_background_opacity": "0.2",
            "related_modal_rounded_corners": True,
            "related_modal_close_button_visible": True,
            "list_filter_highlight": True,
            "list_filter_dropdown": True,
            "list_filter_sticky": True,
            "list_filter_removal_links": True,
            "foldable_apps": True,
            "show_fieldsets_as_tabs": False,
            "show_inlines_as_tabs": False,
            "collapsible_stacked_inlines": False,
            "collapsible_tabular_inlines": False,
            "recent_actions_visible": True,
            "form_actions_sticky": True,
            "form_submit_sticky": True,
            "form_pagination_sticky": True,
        }

        dark_theme = {
            "name": "Bunoraa Dark",
            "title": "Bunoraa Administration",
            "title_visible": True,
            "logo_visible": True,
            "logo_color": "#f9b939",
            "title_color": "#f9b939",
            "env_color": "#f9b939",
            "env_visible_in_header": True,
            "env_visible_in_favicon": True,
            "language_chooser_active": True,
            "language_chooser_control": "minimal-select",
            "language_chooser_display": "name",
            "css_header_background_color": "#1a1a1a",
            "css_header_text_color": "#fafafa",
            "css_header_link_color": "#f9b939",
            "css_header_link_hover_color": "#f7a708",
            "css_module_background_color": "#c35322",
            "css_module_background_selected_color": "#292524",
            "css_module_text_color": "#ffffff",
            "css_module_link_color": "#ffffff",
            "css_module_link_selected_color": "#ffffff",
            "css_module_link_hover_color": "#fde68a",
            "css_module_rounded_corners": True,
            "css_generic_link_color": "#f9b939",
            "css_generic_link_hover_color": "#f7a708",
            "css_generic_link_active_color": "#d97706",
            "css_save_button_background_color": "#c35322",
            "css_save_button_background_hover_color": "#af3625",
            "css_save_button_text_color": "#ffffff",
            "css_delete_button_background_color": "#ef4444",
            "css_delete_button_background_hover_color": "#dc2626",
            "css_delete_button_text_color": "#ffffff",
            "related_modal_active": True,
            "related_modal_background_color": "#0f0f0f",
            "related_modal_background_opacity": "0.3",
            "related_modal_rounded_corners": True,
            "related_modal_close_button_visible": True,
            "list_filter_highlight": True,
            "list_filter_dropdown": True,
            "list_filter_sticky": True,
            "list_filter_removal_links": True,
            "foldable_apps": True,
            "show_fieldsets_as_tabs": False,
            "show_inlines_as_tabs": False,
            "collapsible_stacked_inlines": False,
            "collapsible_tabular_inlines": False,
            "recent_actions_visible": True,
            "form_actions_sticky": True,
            "form_submit_sticky": True,
            "form_pagination_sticky": True,
        }

        active_exists = Theme.objects.filter(active=True).exists()
        created = 0
        updated = 0

        for theme_values in (light_theme, dark_theme):
            theme, was_created = Theme.objects.get_or_create(
                name=theme_values["name"], defaults={**theme_values, "active": False}
            )
            if was_created:
                created += 1
            else:
                for key, value in theme_values.items():
                    setattr(theme, key, value)
                theme.save()
                updated += 1

        if not active_exists:
            Theme.objects.filter(name=light_theme["name"]).update(active=True)

        self.stdout.write(
            self.style.SUCCESS(
                f"Admin interface themes synced. Created: {created}, Updated: {updated}."
            )
        )
