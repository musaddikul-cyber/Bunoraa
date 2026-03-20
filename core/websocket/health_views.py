"""
WebSocket Health & Metrics API Views
"""
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAdminUser
from rest_framework.response import Response
from rest_framework import status
from core.websocket.monitoring import WebSocketHealthCheck, WebSocketMetrics
import logging

logger = logging.getLogger('bunoraa.websocket')


@api_view(['GET'])
@permission_classes([AllowAny])
def websocket_health(request):
    """
    WebSocket system health check endpoint.
    
    Returns:
    {
        'status': 'healthy|degraded|unhealthy',
        'metrics': {
            'active_connections': int,
            'active_users': int,
            'total_connections_lifetime': int,
            'timestamp': float
        },
        'issues': [
            {
                'severity': 'warning|error',
                'message': str
            }
        ]
    }
    """
    try:
        health = WebSocketHealthCheck.check_health()
        
        return Response({
            'success': True,
            'data': health,
        }, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"WebSocket health check error: {e}", exc_info=True)
        return Response({
            'success': False,
            'error': 'Health check failed',
            'message': str(e),
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAdminUser])
def websocket_metrics(request):
    """
    Get detailed WebSocket metrics (admin only).
    
    Returns:
    {
        'active_connections': int,
        'active_users': int,
        'total_connections_lifetime': int,
        'timestamp': float
    }
    """
    try:
        metrics = WebSocketMetrics.get_metrics()
        
        return Response({
            'success': True,
            'data': metrics,
        }, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"WebSocket metrics error: {e}", exc_info=True)
        return Response({
            'success': False,
            'error': 'Metrics retrieval failed',
            'message': str(e),
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAdminUser])
def websocket_status(request):
    """
    Simple WebSocket status check (admin only).
    Useful for monitoring and alerting.
    
    Returns:
    {
        'status': 'up|down',
        'message': str
    }
    """
    try:
        health = WebSocketHealthCheck.check_health()
        is_healthy = health['status'] in ('healthy', 'degraded')
        
        return Response({
            'status': 'up' if is_healthy else 'down',
            'message': f"WebSocket system is {health['status']}",
            'system_status': health['status'],
            'active_connections': health['metrics'].get('active_connections', 0),
        }, status=status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE)
    except Exception as e:
        logger.error(f"WebSocket status error: {e}")
        return Response({
            'status': 'down',
            'message': 'WebSocket system unavailable',
            'error': str(e),
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
