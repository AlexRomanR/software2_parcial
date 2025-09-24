# from django.shortcuts import render
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.db import connection
from ingestion.models import DataSource
from ingestion.services import get_schema_info
from .models import KPI, AlertRule, Alert
from .services import AlertManager, KPICalculator
from .forms import KPIForm, AlertRuleForm

@login_required
def alerts_dashboard(request):
    """Dashboard principal de alertas"""
    user_alerts = AlertManager.get_user_alerts(request.user)
    alert_summary = AlertManager.get_alert_summary(request.user)
    
    # KPIs del usuario
    user_kpis = KPI.objects.filter(owner=request.user, is_active=True)
    
    context = {
        'alerts': user_alerts[:10],  # Últimas 10 alertas
        'alert_summary': alert_summary,
        'user_kpis': user_kpis
    }
    return render(request, 'notifications/dashboard.html', context)

@login_required
def create_kpi(request):
    """Crear un nuevo KPI"""
    if request.method == 'POST':
        form = KPIForm(request.POST, user=request.user)
        if form.is_valid():
            kpi = form.save(commit=False)
            kpi.owner = request.user
            kpi.save()
            messages.success(request, f'KPI "{kpi.name}" creado exitosamente.')
            return redirect('notifications:dashboard')
    else:
        form = KPIForm(user=request.user)
    
    return render(request, 'notifications/create_kpi.html', {'form': form})

@login_required
def create_alert_rule(request, kpi_id=None):
    """Crear una nueva regla de alerta"""
    kpi = None
    if kpi_id:
        kpi = get_object_or_404(KPI, id=kpi_id, owner=request.user)
    
    if request.method == 'POST':
        form = AlertRuleForm(request.POST, user=request.user, initial_kpi=kpi)
        if form.is_valid():
            alert_rule = form.save()
            messages.success(request, f'Regla de alerta "{alert_rule.name}" creada exitosamente.')
            return redirect('notifications:dashboard')
    else:
        form = AlertRuleForm(user=request.user, initial_kpi=kpi)
    
    return render(request, 'notifications/create_alert_rule.html', {'form': form, 'kpi': kpi})

@login_required
def get_columns_ajax(request):
    """AJAX: Obtener columnas de una fuente de datos"""
    data_source_id = request.GET.get('data_source_id')
    if not data_source_id:
        return JsonResponse({'columns': []})
    
    try:
        data_source = DataSource.objects.get(id=data_source_id, owner=request.user)
        schema_info = get_schema_info(data_source.internal_schema)
        
        # Obtener todas las columnas de todas las tablas
        columns = []
        for table, info in schema_info.items():
            for col in info['columns']:
                columns.append(col)
        
        return JsonResponse({'columns': list(set(columns))})  # Eliminar duplicados
    except DataSource.DoesNotExist:
        return JsonResponse({'columns': []})

@require_POST
@login_required
def acknowledge_alert(request, alert_id):
    """Reconocer una alerta"""
    success = AlertManager.acknowledge_alert(alert_id, request.user)
    if success:
        messages.success(request, 'Alerta reconocida.')
    else:
        messages.error(request, 'No se pudo reconocer la alerta.')
    return redirect('notifications:dashboard')

@require_POST
@login_required
def resolve_alert(request, alert_id):
    """Resolver una alerta"""
    success = AlertManager.resolve_alert(alert_id, request.user)
    if success:
        messages.success(request, 'Alerta marcada como resuelta.')
    else:
        messages.error(request, 'No se pudo resolver la alerta.')
    return redirect('notifications:dashboard')

@login_required
def test_kpi(request, kpi_id):
    """Probar cálculo de un KPI"""
    kpi = get_object_or_404(KPI, id=kpi_id, owner=request.user)
    
    try:
        value = KPICalculator.calculate_kpi_value(kpi)
        messages.success(request, f'KPI "{kpi.name}" calculado: {value}')
    except Exception as e:
        messages.error(request, f'Error calculando KPI: {str(e)}')
    
    return redirect('notifications:dashboard')

@login_required
def list_kpis(request):
    """Lista todos los KPIs del usuario"""
    kpis = KPI.objects.filter(owner=request.user).order_by('-created_at')
    return render(request, 'notifications/list_kpis.html', {'kpis': kpis})

@login_required
def list_alert_rules(request):
    """Lista todas las reglas de alerta del usuario"""
    rules = AlertRule.objects.filter(kpi__owner=request.user).order_by('-created_at')
    return render(request, 'notifications/list_alert_rules.html', {'rules': rules})

@require_POST
@login_required
def delete_kpi(request, kpi_id):
    """Eliminar un KPI y sus reglas de alerta asociadas"""
    kpi = get_object_or_404(KPI, id=kpi_id, owner=request.user)
    
    # Contar cuántas reglas de alerta se eliminarán
    alert_rules_count = AlertRule.objects.filter(kpi=kpi).count()
    
    kpi_name = kpi.name
    kpi.delete()  # Esto eliminará en cascada las reglas de alerta y alertas
    
    messages.success(
        request, 
        f'KPI "{kpi_name}" eliminado exitosamente. '
        f'Se eliminaron {alert_rules_count} regla(s) de alerta asociada(s).'
    )
    return redirect('notifications:list_kpis')

@require_POST
@login_required
def delete_alert_rule(request, rule_id):
    """Eliminar una regla de alerta"""
    alert_rule = get_object_or_404(AlertRule, id=rule_id, kpi__owner=request.user)
    
    # Contar cuántas alertas activas se resolverán
    active_alerts = Alert.objects.filter(
        alert_rule=alert_rule, 
        status__in=['new', 'acknowledged']
    ).count()
    
    rule_name = alert_rule.name
    alert_rule.delete()  # Esto eliminará en cascada las alertas
    
    messages.success(
        request, 
        f'Regla de alerta "{rule_name}" eliminada exitosamente. '
        f'Se resolvieron {active_alerts} alerta(s) activa(s).'
    )
    return redirect('notifications:list_alert_rules')

@require_POST
@login_required
def delete_alert(request, alert_id):
    """Eliminar/descartar una alerta específica"""
    alert = get_object_or_404(Alert, id=alert_id, alert_rule__kpi__owner=request.user)
    
    alert.status = 'dismissed'
    alert.resolved_by = request.user
    alert.resolved_at = timezone.now()
    alert.save()
    
    messages.success(request, 'Alerta descartada exitosamente.')
    return redirect('notifications:dashboard')