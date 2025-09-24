from decimal import Decimal
from django.core.mail import send_mail
from django.conf import settings
from django.db import connection
from django.utils import timezone
from typing import Dict, List, Any
import logging

from .models import KPI, AlertRule, Alert, NotificationLog

logger = logging.getLogger(__name__)

class KPICalculator:
    """Servicio para calcular valores de KPIs"""
    
    @staticmethod
    def calculate_kpi_value(kpi: KPI) -> Decimal:
        """Calcula el valor actual de un KPI"""
        try:
            schema = kpi.data_source.internal_schema
            table = kpi.data_source.internal_table
            
            if not schema or not table:
                raise ValueError(f"KPI {kpi.name}: datos no disponibles")
            
            # Construir la consulta SQL según el tipo de métrica
            column = kpi.column_name
            metric_type = kpi.metric_type
            
            # Funciones SQL según el tipo de métrica
            sql_functions = {
                'sum': f'SUM({column})',
                'avg': f'AVG({column})',
                'count': f'COUNT({column})',
                'max': f'MAX({column})',
                'min': f'MIN({column})',
            }
            
            if metric_type not in sql_functions:
                raise ValueError(f"Tipo de métrica no válido: {metric_type}")
            
            sql_function = sql_functions[metric_type]
            base_query = f'SELECT {sql_function} as result FROM "{schema}"."{table}"'
            
            # Aplicar filtros si existen
            where_conditions = []
            if kpi.filter_conditions:
                for field, value in kpi.filter_conditions.items():
                    if isinstance(value, str):
                        where_conditions.append(f"{field} = '{value}'")
                    else:
                        where_conditions.append(f"{field} = {value}")
            
            if where_conditions:
                base_query += " WHERE " + " AND ".join(where_conditions)
            
            # Ejecutar consulta
            with connection.cursor() as cursor:
                cursor.execute(base_query)
                result = cursor.fetchone()
                return Decimal(str(result[0] or 0))
                
        except Exception as e:
            logger.error(f"Error calculando KPI {kpi.name}: {str(e)}")
            raise

class AlertEvaluator:
    """Servicio para evaluar reglas de alertas"""
    
    @staticmethod
    def evaluate_alert_rule(alert_rule: AlertRule) -> bool:
        """Evalúa si una regla de alerta debe dispararse"""
        try:
            current_value = KPICalculator.calculate_kpi_value(alert_rule.kpi)
            threshold = alert_rule.threshold_value
            operator = alert_rule.comparison_operator
            
            # Evaluar condición según el operador
            conditions = {
                'gt': current_value > threshold,
                'gte': current_value >= threshold,
                'lt': current_value < threshold,
                'lte': current_value <= threshold,
                'eq': current_value == threshold,
                'ne': current_value != threshold,
            }
            
            should_alert = conditions.get(operator, False)
            
            if should_alert:
                # Crear la alerta
                AlertEvaluator._create_alert(alert_rule, current_value, threshold)
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error evaluando alerta {alert_rule.name}: {str(e)}")
            return False
    
    @staticmethod
    def _create_alert(alert_rule: AlertRule, current_value: Decimal, threshold_value: Decimal):
        """Crea una nueva alerta"""
        # Verificar si ya existe una alerta activa para esta regla
        existing_alert = Alert.objects.filter(
            alert_rule=alert_rule,
            status__in=['new', 'acknowledged']
        ).first()
        
        if existing_alert:
            # Actualizar valor actual
            existing_alert.current_value = current_value
            existing_alert.save()
            return existing_alert
        
        # Crear nueva alerta
        message = (f"KPI '{alert_rule.kpi.name}' ha superado el umbral. "
                  f"Valor actual: {current_value}, Umbral: {threshold_value}")
        
        alert = Alert.objects.create(
            alert_rule=alert_rule,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            context_data={
                'kpi_name': alert_rule.kpi.name,
                'metric_type': alert_rule.kpi.metric_type,
                'column_name': alert_rule.kpi.column_name,
                'severity': alert_rule.severity,
            }
        )
        
        # Enviar notificaciones
        NotificationService.send_alert_notifications(alert)
        return alert
    
    @staticmethod
    def check_all_active_rules():
        """Verifica todas las reglas de alerta activas"""
        active_rules = AlertRule.objects.filter(is_active=True)
        triggered_count = 0
        
        for rule in active_rules:
            try:
                if AlertEvaluator.evaluate_alert_rule(rule):
                    triggered_count += 1
            except Exception as e:
                logger.error(f"Error procesando regla {rule.name}: {str(e)}")
        
        return triggered_count

class NotificationService:
    """Servicio para envío de notificaciones"""
    
    @staticmethod
    def send_alert_notifications(alert: Alert):
        """Envía todas las notificaciones configuradas para una alerta"""
        alert_rule = alert.alert_rule
        
        # Notificación por email
        if alert_rule.send_email:
            NotificationService._send_email_notification(alert)
        
        # Notificación en la app (se registra para mostrar en el dashboard)
        if alert_rule.send_in_app:
            NotificationService._log_in_app_notification(alert)
    
    @staticmethod
    def _send_email_notification(alert: Alert):
        """Envía notificación por email"""
        try:
            alert_rule = alert.alert_rule
            recipients = alert_rule.email_recipients or [alert_rule.kpi.owner.email]
            
            subject = f"🚨 Alerta {alert_rule.severity.upper()}: {alert_rule.name}"
            
            message = f"""
Estimado usuario,

Se ha disparado una alerta en el sistema BI:

KPI: {alert_rule.kpi.name}
Regla: {alert_rule.name}
Severidad: {alert_rule.get_severity_display()}

Valor actual: {alert.current_value}
Umbral configurado: {alert.threshold_value}
Condición: {alert_rule.get_comparison_operator_display()}

Descripción: {alert.message}

Fecha y hora: {alert.triggered_at.strftime('%d/%m/%Y %H:%M:%S')}

Por favor, revise el dashboard para más detalles.

---
Sistema BI - Alertas Automáticas
"""
            
            success = send_mail(
                subject=subject,
                message=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=recipients,
                fail_silently=False,
            )
            
            # Log de la notificación
            for recipient in recipients:
                NotificationLog.objects.create(
                    alert=alert,
                    notification_type='email',
                    recipient=recipient,
                    success=success
                )
                
        except Exception as e:
            logger.error(f"Error enviando email para alerta {alert.id}: {str(e)}")
            # Log del error
            NotificationLog.objects.create(
                alert=alert,
                notification_type='email',
                recipient='error',
                success=False,
                error_message=str(e)
            )
    
    @staticmethod
    def _log_in_app_notification(alert: Alert):
        """Registra notificación en la aplicación"""
        NotificationLog.objects.create(
            alert=alert,
            notification_type='in_app',
            recipient=alert.alert_rule.kpi.owner.username,
            success=True
        )

class AlertManager:
    """Servicio principal para gestión de alertas"""
    
    @staticmethod
    def acknowledge_alert(alert_id: str, user):
        """Marca una alerta como reconocida"""
        try:
            alert = Alert.objects.get(id=alert_id)
            alert.status = 'acknowledged'
            alert.acknowledged_by = user
            alert.acknowledged_at = timezone.now()
            alert.save()
            return True
        except Alert.DoesNotExist:
            return False
    
    @staticmethod
    def resolve_alert(alert_id: str, user):
        """Marca una alerta como resuelta"""
        try:
            alert = Alert.objects.get(id=alert_id)
            alert.status = 'resolved'
            alert.resolved_by = user
            alert.resolved_at = timezone.now()
            alert.save()
            return True
        except Alert.DoesNotExist:
            return False
    
    @staticmethod
    def get_user_alerts(user, status=None) -> List[Alert]:
        """Obtiene las alertas de un usuario"""
        queryset = Alert.objects.filter(alert_rule__kpi__owner=user)
        
        if status:
            queryset = queryset.filter(status=status)
        
        return queryset.order_by('-triggered_at')
    
    @staticmethod
    def get_alert_summary(user) -> Dict[str, Any]:
        """Obtiene resumen de alertas para el dashboard"""
        alerts = Alert.objects.filter(alert_rule__kpi__owner=user)
        
        summary = {
            'total': alerts.count(),
            'new': alerts.filter(status='new').count(),
            'acknowledged': alerts.filter(status='acknowledged').count(),
            'resolved': alerts.filter(status='resolved').count(),
            'critical': alerts.filter(
                alert_rule__severity='critical',
                status__in=['new', 'acknowledged']
            ).count(),
            'recent': alerts.filter(
                status__in=['new', 'acknowledged']
            )[:5]  # Últimas 5 alertas activas
        }
        
        return summary