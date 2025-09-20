from django.db import models
from django.contrib.auth import get_user_model
from fernet_fields import EncryptedTextField, EncryptedCharField
from django.contrib.auth.models import User
import pandas as pd
from .services import get_engine

User = get_user_model()

class DataSource(models.Model):
    FILE = "FILE"
    LIVE = "LIVE"
    TYPE_CHOICES = [(FILE, "Archivo importado"), (LIVE, "Conexión en vivo")]

    name = models.CharField(max_length=120)
    kind = models.CharField(max_length=10, choices=TYPE_CHOICES, default=FILE)
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    # Para datasets importados: guardamos a qué esquema/tabla interna fue cargado
    internal_schema = models.CharField(max_length=120, blank=True, default="")
    internal_table = models.CharField(max_length=120, blank=True, default="")

    def __str__(self):
        return f"{self.name} ({self.kind})"
    
    @property
    def has_dashboards(self):
        """Verifica si este archivo tiene dashboards asociados"""
        return self.dashboards.exists()
    
    @property
    def auto_dashboard(self):
        """Obtiene o crea el dashboard automático para este archivo"""
        dashboard, created = Dashboard.objects.get_or_create(
            data_source=self,
            is_auto_generated=True,
            defaults={
                'name': f'Dashboard - {self.name}',
                'description': f'Dashboard automático generado para {self.name}',
                'owner': self.owner
            }
        )
        return dashboard
    
    def get_column_info(self):
        """Obtiene información detallada de las columnas"""
        if not self.internal_schema or not self.internal_table:
            return {}
        
        try:
            engine = get_engine()
            with engine.begin() as conn:
                # Obtener tipos de columnas
                result = conn.execute(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_schema = '{self.internal_schema}' 
                    AND table_name = '{self.internal_table}'
                    ORDER BY ordinal_position
                """)
                
                columns = {}
                for row in result:
                    col_name, data_type, is_nullable = row
                    columns[col_name] = {
                        'type': data_type,
                        'nullable': is_nullable == 'YES',
                        'suggested_chart': self._suggest_chart_type(data_type)
                    }
                return columns
        except Exception:
            return {}
    
    def _suggest_chart_type(self, data_type):
        """Sugiere tipo de gráfico basado en el tipo de dato"""
        if data_type in ['integer', 'bigint', 'numeric', 'real', 'double precision']:
            return 'bar'
        elif data_type in ['date', 'timestamp', 'timestamp with time zone']:
            return 'line'
        elif data_type in ['text', 'varchar', 'character varying']:
            return 'pie'
        return 'bar'

class UploadedDataset(models.Model):
    CSV = "csv"
    XLSX = "xlsx"
    SQL = "sql"
    FILE_TYPES = [(CSV, "CSV"), (XLSX, "Excel"), (SQL, "SQL")]

    source = models.OneToOneField(DataSource, on_delete=models.CASCADE, related_name="upload")
    file = models.FileField(upload_to="uploads/")
    file_type = models.CharField(max_length=10, choices=FILE_TYPES)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    rows_ingested = models.PositiveIntegerField(default=0)
    columns = models.JSONField(default=list, blank=True)
    
    def get_sample_data(self, limit=5):
        """Obtiene datos de muestra para preview"""
        if not self.source.internal_schema or not self.source.internal_table:
            return []
        
        try:
            engine = get_engine()
            query = f'SELECT * FROM "{self.source.internal_schema}"."{self.source.internal_table}" LIMIT {limit}'
            df = pd.read_sql(query, engine)
            return df.to_dict('records')
        except Exception:
            return []

class ExternalConnection(models.Model):
    POSTGRES = "postgres"
    MYSQL = "mysql"
    DB_TYPES = [(POSTGRES, "PostgreSQL"), (MYSQL, "MySQL/MariaDB")]

    source = models.OneToOneField(DataSource, on_delete=models.CASCADE, related_name="connection")
    db_type = models.CharField(max_length=20, choices=DB_TYPES, default=POSTGRES)
    host = models.CharField(max_length=255)
    port = models.IntegerField(default=5432)
    database = models.CharField(max_length=255)
    username = EncryptedCharField(max_length=255)
    password = EncryptedTextField()
    extras = models.JSONField(default=dict, blank=True)   # sslmode, options, etc.
    created_at = models.DateTimeField(auto_now_add=True)

# Modelos para dashboards mejorados

class Dashboard(models.Model):
    name = models.CharField(max_length=120)
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # NUEVO: Relación con archivo de datos
    data_source = models.ForeignKey(DataSource, on_delete=models.CASCADE, related_name="dashboards", null=True, blank=True)
    is_auto_generated = models.BooleanField(default=False, help_text="Dashboard generado automáticamente")
    
    # Configuración de layout
    grid_columns = models.PositiveIntegerField(default=12, help_text="Columnas del grid (1-12)")
    
    class Meta:
        ordering = ["-created_at"]
        
    def __str__(self):
        return self.name
    
    @property
    def widget_count(self):
        return self.widgets.count()
    
    @property 
    def kpi_count(self):
        return self.kpis.count()
    
    def generate_auto_widgets(self):
        """Genera widgets automáticamente basado en los datos del archivo"""
        if not self.data_source or not self.data_source.internal_schema:
            return
        
        columns = self.data_source.get_column_info()
        schema = self.data_source.internal_schema
        table = self.data_source.internal_table
        
        order = 0
        
        # Generar KPIs automáticos
        numeric_columns = [col for col, info in columns.items() 
                          if info['type'] in ['integer', 'bigint', 'numeric', 'real', 'double precision']]
        
        for col in numeric_columns[:3]:  # Máximo 3 KPIs
            KpiCard.objects.get_or_create(
                dashboard=self,
                title=f'Total {col.title()}',
                defaults={
                    'sql': f'SELECT SUM("{col}") as value FROM "{schema}"."{table}"',
                    'order': order,
                    'suffix': ''
                }
            )
            order += 1
        
        # Generar widgets automáticos
        text_columns = [col for col, info in columns.items() 
                       if info['type'] in ['text', 'varchar', 'character varying']]
        
        for i, text_col in enumerate(text_columns[:2]):  # Máximo 2 gráficos de categorías
            if numeric_columns:
                Widget.objects.get_or_create(
                    dashboard=self,
                    title=f'{text_col.title()} vs {numeric_columns[0].title()}',
                    defaults={
                        'chart_type': 'pie',
                        'sql': f'SELECT "{text_col}" as label, SUM("{numeric_columns[0]}") as value FROM "{schema}"."{table}" GROUP BY "{text_col}" LIMIT 10',
                        'order': order
                    }
                )
                order += 1
        
        # Gráfico temporal si hay columnas de fecha
        date_columns = [col for col, info in columns.items() 
                       if 'date' in info['type'] or 'timestamp' in info['type']]
        
        if date_columns and numeric_columns:
            Widget.objects.get_or_create(
                dashboard=self,
                title=f'Tendencia temporal - {numeric_columns[0].title()}',
                defaults={
                    'chart_type': 'line',
                    'sql': f'SELECT DATE("{date_columns[0]}") as label, SUM("{numeric_columns[0]}") as value FROM "{schema}"."{table}" GROUP BY DATE("{date_columns[0]}") ORDER BY label',
                    'order': order
                }
            )

class Widget(models.Model):
    BAR="bar"; LINE="line"; PIE="pie"; DOUGHNUT="doughnut"; SCATTER="scatter"; RADAR="radar"; AREA="area"
    TYPES = [
        (BAR,"Barras"), (LINE,"Líneas"), (PIE,"Circular"), 
        (DOUGHNUT,"Dona"), (SCATTER,"Dispersión"), (RADAR,"Radar"), (AREA,"Área")
    ]
    
    dashboard = models.ForeignKey(Dashboard, related_name="widgets", on_delete=models.CASCADE)
    title = models.CharField(max_length=160)
    chart_type = models.CharField(max_length=12, choices=TYPES, default=BAR)
    sql = models.TextField(help_text="SELECT label, value FROM ...")
    order = models.PositiveIntegerField(default=0)
    
    # NUEVO: Configuración de layout
    width = models.PositiveIntegerField(default=6, help_text="Ancho en columnas (1-12)")
    height = models.PositiveIntegerField(default=400, help_text="Altura en pixels")
    
    # NUEVO: Configuración visual
    color_scheme = models.CharField(max_length=20, default="default", 
                                   choices=[("default", "Por defecto"), ("blue", "Azul"), 
                                           ("green", "Verde"), ("red", "Rojo"), ("purple", "Morado")])
    
    class Meta:
        ordering = ["order","id"]
        
    def __str__(self):
        return f"{self.dashboard.name} · {self.title}"
    
    def get_chart_colors(self):
        """Retorna colores basados en el esquema seleccionado"""
        color_schemes = {
            "default": ["#36A2EB", "#FF6384", "#4BC0C0", "#FF9F40", "#9966FF"],
            "blue": ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c"],
            "green": ["#2ca02c", "#98df8a", "#d62728", "#ff9896", "#ff7f0e"],
            "red": ["#d62728", "#ff9896", "#2ca02c", "#98df8a", "#ff7f0e"],
            "purple": ["#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2"]
        }
        return color_schemes.get(self.color_scheme, color_schemes["default"])

class KpiCard(models.Model):
    dashboard = models.ForeignKey(Dashboard, related_name="kpis", on_delete=models.CASCADE)
    title = models.CharField(max_length=120)
    sql = models.TextField(help_text="SELECT value FROM ...")
    order = models.PositiveIntegerField(default=0)
    suffix = models.CharField(max_length=32, blank=True, default="")
    
    # NUEVO: Configuración visual
    icon = models.CharField(max_length=50, blank=True, default="", help_text="Clase de icono (ej: fas fa-dollar-sign)")
    color = models.CharField(max_length=20, default="primary", 
                           choices=[("primary", "Azul"), ("success", "Verde"), 
                                   ("warning", "Amarillo"), ("danger", "Rojo"), ("info", "Cyan")])
    
    class Meta:
        ordering = ["order","id"]
        
    def __str__(self):
        return f"{self.dashboard.name} · {self.title}"

class SavedQuery(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=160)
    sql = models.TextField()
    is_approved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # NUEVO: Relación con archivo
    data_source = models.ForeignKey(DataSource, on_delete=models.CASCADE, null=True, blank=True)
    
    class Meta:
        ordering = ["-created_at"]
        
    def __str__(self):
        return self.title

# NUEVO: Modelo para análisis automático de datos
class DataAnalysis(models.Model):
    data_source = models.OneToOneField(DataSource, on_delete=models.CASCADE, related_name="analysis")
    
    # Estadísticas generales
    total_rows = models.PositiveIntegerField(default=0)
    numeric_columns = models.JSONField(default=list)
    text_columns = models.JSONField(default=list) 
    date_columns = models.JSONField(default=list)
    
    # Sugerencias de visualización
    suggested_charts = models.JSONField(default=list)
    
    # Metadatos
    analyzed_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Análisis de {self.data_source.name}"