"""
Management command to check table schema.
"""
from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = 'Check table columns and structure'

    def add_arguments(self, parser):
        parser.add_argument(
            '--table',
            type=str,
            help='Specific table name to check',
        )

    def handle(self, *args, **options):
        table_name = options.get('table')
        
        if table_name:
            self.check_table(table_name)
        else:
            # Check all relevant tables
            tables = [
                'catalog_product',
                'analytics_pageview',
                'orders_order',
                'accounts_userbehaviorprofile',
                'accounts_userinteraction',
            ]
            for table in tables:
                self.check_table(table)

    def check_table(self, table_name):
        """Check columns in a table."""
        self.stdout.write(f'\n{table_name}:')
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = %s 
                    ORDER BY ordinal_position
                """, [table_name])
                
                rows = cursor.fetchall()
                if rows:
                    for col_name, col_type, nullable in rows:
                        null_str = "NULL" if nullable == "YES" else "NOT NULL"
                        self.stdout.write(f'  {col_name:30} {col_type:20} {null_str}')
                else:
                    self.stdout.write(self.style.ERROR(f'  [TABLE NOT FOUND]'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'  [ERROR] {e}'))
