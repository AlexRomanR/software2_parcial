from django.urls import path
from . import views

app_name = "notifications"

urlpatterns = [
    # Dashboard principal de alertas
    path('', views.alerts_dashboard, name='dashboard'),
    
    # Gestión de KPIs
    path('kpis/', views.list_kpis, name='list_kpis'),
    path('kpi/create/', views.create_kpi, name='create_kpi'),
    path('kpi/<int:kpi_id>/test/', views.test_kpi, name='test_kpi'),
    
    # Gestión de reglas de alerta
    path('rules/', views.list_alert_rules, name='list_alert_rules'),
    path('rule/create/', views.create_alert_rule, name='create_alert_rule'),
    path('rule/create/<int:kpi_id>/', views.create_alert_rule, name='create_alert_rule_for_kpi'),
    
    # Acciones sobre alertas
    path('alert/<uuid:alert_id>/acknowledge/', views.acknowledge_alert, name='acknowledge_alert'),
    path('alert/<uuid:alert_id>/resolve/', views.resolve_alert, name='resolve_alert'),
    
    # AJAX endpoints
    path('ajax/columns/', views.get_columns_ajax, name='get_columns_ajax'),
]