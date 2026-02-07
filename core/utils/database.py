"""
Database optimization utilities for PostgreSQL.
Includes index management, query optimization, and maintenance tasks.
"""
import logging
from typing import List, Dict, Optional
from django.db import connection
from django.conf import settings

logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """
    PostgreSQL database optimization utilities.
    """
    
    def __init__(self):
        self.is_postgres = 'postgresql' in settings.DATABASES['default']['ENGINE']
    
    def analyze_tables(self, tables: Optional[List[str]] = None) -> Dict:
        """
        Run ANALYZE on specified tables or all tables.
        Updates statistics for the query planner.
        """
        if not self.is_postgres:
            return {'status': 'skipped', 'reason': 'Not PostgreSQL'}
        
        results = {'analyzed': [], 'errors': []}
        
        with connection.cursor() as cursor:
            if tables is None:
                # Get all table names
                cursor.execute("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                """)
                tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                try:
                    cursor.execute(f'ANALYZE "{table}"')
                    results['analyzed'].append(table)
                except Exception as e:
                    results['errors'].append({'table': table, 'error': str(e)})
                    logger.error(f'Failed to analyze {table}: {e}')
        
        return results
    
    def vacuum_tables(self, tables: Optional[List[str]] = None, 
                      full: bool = False, analyze: bool = True) -> Dict:
        """
        Run VACUUM on specified tables or all tables.
        
        Args:
            tables: List of table names (None for all)
            full: Run VACUUM FULL (reclaims more space but locks tables)
            analyze: Also run ANALYZE
        """
        if not self.is_postgres:
            return {'status': 'skipped', 'reason': 'Not PostgreSQL'}
        
        results = {'vacuumed': [], 'errors': []}
        
        # VACUUM cannot run in a transaction
        old_autocommit = connection.connection.autocommit
        connection.connection.autocommit = True
        
        try:
            with connection.cursor() as cursor:
                if tables is None:
                    cursor.execute("""
                        SELECT tablename 
                        FROM pg_tables 
                        WHERE schemaname = 'public'
                    """)
                    tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    try:
                        vacuum_type = 'FULL ANALYZE' if full else ('ANALYZE' if analyze else '')
                        cursor.execute(f'VACUUM {vacuum_type} "{table}"')
                        results['vacuumed'].append(table)
                    except Exception as e:
                        results['errors'].append({'table': table, 'error': str(e)})
                        logger.error(f'Failed to vacuum {table}: {e}')
        finally:
            connection.connection.autocommit = old_autocommit
        
        return results
    
    def get_table_stats(self) -> List[Dict]:
        """
        Get table statistics including row counts and sizes.
        """
        if not self.is_postgres:
            return []
        
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    relname AS table_name,
                    n_live_tup AS row_count,
                    pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
                    pg_size_pretty(pg_relation_size(relid)) AS table_size,
                    pg_size_pretty(pg_indexes_size(relid)) AS index_size,
                    n_dead_tup AS dead_rows,
                    last_vacuum,
                    last_analyze
                FROM pg_stat_user_tables
                ORDER BY pg_total_relation_size(relid) DESC
            """)
            
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_missing_indexes(self) -> List[Dict]:
        """
        Find potential missing indexes based on query patterns.
        """
        if not self.is_postgres:
            return []
        
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    schemaname || '.' || relname AS table,
                    seq_scan - idx_scan AS too_much_seq,
                    CASE 
                        WHEN seq_scan - idx_scan > 0 
                        THEN 'Missing Index?' 
                        ELSE 'OK' 
                    END AS status,
                    pg_size_pretty(pg_relation_size(relid)) AS table_size,
                    seq_scan,
                    idx_scan
                FROM pg_stat_user_tables
                WHERE seq_scan - idx_scan > 0
                AND pg_relation_size(relid) > 1000000  -- Only tables > 1MB
                ORDER BY too_much_seq DESC
                LIMIT 20
            """)
            
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_unused_indexes(self) -> List[Dict]:
        """
        Find indexes that are not being used.
        """
        if not self.is_postgres:
            return []
        
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    schemaname || '.' || relname AS table,
                    indexrelname AS index,
                    pg_size_pretty(pg_relation_size(indexrelid)) AS size,
                    idx_scan AS index_scans
                FROM pg_stat_user_indexes
                WHERE idx_scan = 0
                AND schemaname = 'public'
                AND indexrelname NOT LIKE '%_pkey'
                ORDER BY pg_relation_size(indexrelid) DESC
            """)
            
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_slow_queries(self, limit: int = 20) -> List[Dict]:
        """
        Get slowest queries from pg_stat_statements if available.
        """
        if not self.is_postgres:
            return []
        
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        substring(query, 1, 100) AS query_preview,
                        calls,
                        round(total_exec_time::numeric, 2) AS total_time_ms,
                        round(mean_exec_time::numeric, 2) AS avg_time_ms,
                        round((100 * total_exec_time / sum(total_exec_time) 
                            OVER ())::numeric, 2) AS percentage
                    FROM pg_stat_statements
                    ORDER BY total_exec_time DESC
                    LIMIT %s
                """, [limit])
                
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception:
            # pg_stat_statements extension not available
            return []
    
    def get_connection_stats(self) -> Dict:
        """
        Get current connection statistics.
        """
        if not self.is_postgres:
            return {}
        
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    count(*) AS total_connections,
                    count(*) FILTER (WHERE state = 'active') AS active,
                    count(*) FILTER (WHERE state = 'idle') AS idle,
                    count(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_transaction,
                    max(now() - backend_start) AS longest_connection,
                    max(now() - query_start) FILTER (WHERE state = 'active') AS longest_query
                FROM pg_stat_activity
                WHERE datname = current_database()
            """)
            
            row = cursor.fetchone()
            return {
                'total_connections': row[0],
                'active': row[1],
                'idle': row[2],
                'idle_in_transaction': row[3],
                'longest_connection': str(row[4]) if row[4] else None,
                'longest_query': str(row[5]) if row[5] else None,
            }
    
    def create_recommended_indexes(self) -> List[str]:
        """
        Create commonly needed indexes for e-commerce.
        Returns list of created indexes.
        """
        indexes = [
            # Products
            ('catalog_product', 'primary_category_id', 'idx_product_category'),
            ('catalog_product', 'is_active', 'idx_product_active'),
            ('catalog_product', 'created_at', 'idx_product_created'),
            ('catalog_product', 'price', 'idx_product_price'),
            
            # Orders
            ('orders_order', 'user_id', 'idx_order_user'),
            ('orders_order', 'status', 'idx_order_status'),
            ('orders_order', 'created_at', 'idx_order_created'),
            
            # Analytics
            ('analytics_pageview', 'created_at', 'idx_pageview_created'),
            ('analytics_pageview', 'user_id', 'idx_pageview_user'),  # Changed from session_id
            
            # User behavior
            ('accounts_userbehaviorprofile', 'user_id', 'idx_behavior_user'),
            ('accounts_userinteraction', 'user_id', 'idx_interaction_user'),
            ('accounts_userinteraction', 'created_at', 'idx_interaction_created'),
        ]
        
        created = []
        errors = []
        
        with connection.cursor() as cursor:
            # First, get list of existing tables
            cursor.execute("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            existing_tables = {row[0] for row in cursor.fetchall()}
            
            for table, column, index_name in indexes:
                try:
                    # Skip if table doesn't exist
                    if table not in existing_tables:
                        logger.warning(f'Skipping index {index_name}: table "{table}" does not exist')
                        continue
                    
                    # Check if table has the column
                    cursor.execute("""
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = %s AND column_name = %s
                    """, [table, column])
                    
                    if not cursor.fetchone():
                        logger.warning(f'Skipping index {index_name}: column "{column}" does not exist in table "{table}"')
                        continue
                    
                    # Check if index already exists
                    cursor.execute("""
                        SELECT 1 FROM pg_indexes 
                        WHERE indexname = %s
                    """, [index_name])
                    
                    if not cursor.fetchone():
                        cursor.execute(f'''
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS 
                            "{index_name}" ON "{table}" ("{column}")
                        ''')
                        created.append(index_name)
                except Exception as e:
                    error_msg = f'Failed to create index {index_name}: {e}'
                    logger.error(error_msg)
                    errors.append(error_msg)
        
        if errors:
            logger.warning(f'Index creation had {len(errors)} errors (see above)')
        
        return created


def get_database_health() -> Dict:
    """
    Get overall database health status.
    """
    optimizer = DatabaseOptimizer()
    
    if not optimizer.is_postgres:
        return {'status': 'ok', 'database': 'sqlite'}
    
    try:
        stats = optimizer.get_connection_stats()
        table_stats = optimizer.get_table_stats()
        missing_indexes = optimizer.get_missing_indexes()
        unused_indexes = optimizer.get_unused_indexes()
        
        # Calculate health score
        issues = []
        
        if stats.get('idle_in_transaction', 0) > 5:
            issues.append('Too many idle-in-transaction connections')
        
        if len(missing_indexes) > 10:
            issues.append('Multiple tables missing indexes')
        
        # Check for tables needing vacuum
        for table in table_stats:
            if table.get('dead_rows', 0) > 10000:
                issues.append(f"Table {table['table_name']} needs vacuum")
        
        return {
            'status': 'warning' if issues else 'ok',
            'database': 'postgresql',
            'connections': stats,
            'tables_count': len(table_stats),
            'missing_indexes_count': len(missing_indexes),
            'unused_indexes_count': len(unused_indexes),
            'issues': issues,
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }
