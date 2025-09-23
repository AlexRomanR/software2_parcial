from django.db import models
from django.contrib.auth import get_user_model
from fernet_fields import EncryptedTextField, EncryptedCharField
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
    def diagramas_count(self):
        """Retorna cantidad de diagramas generados para este archivo"""
        return self.diagramas.count()

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

# NUEVO: Modelo para guardar diagramas/gráficos generados
class Diagrama(models.Model):
    AUTO = "AUTO"
    CHAT = "CHAT"
    SOURCE_CHOICES = [
        (AUTO, "Generado automáticamente"),
        (CHAT, "Generado por chat")
    ]
    
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    DOUGHNUT = "doughnut"
    SCATTER = "scatter"
    RADAR = "radar"
    AREA = "area"
    
    CHART_TYPES = [
        (BAR, "Barras"),
        (LINE, "Líneas"),
        (PIE, "Circular"),
        (DOUGHNUT, "Dona"),
        (SCATTER, "Dispersión"),
        (RADAR, "Radar"),
        (AREA, "Área")
    ]

    # Relaciones
    data_source = models.ForeignKey(DataSource, on_delete=models.CASCADE, related_name="diagramas")
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Metadatos del diagrama
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True, help_text="Descripción generada por IA")
    chart_type = models.CharField(max_length=20, choices=CHART_TYPES, default=BAR)
    source_type = models.CharField(max_length=10, choices=SOURCE_CHOICES, default=AUTO)
    
    # Datos técnicos
    sql_query = models.TextField(help_text="Consulta SQL que genera los datos")
    chart_data = models.JSONField(help_text="Datos JSON para Chart.js {labels: [], datasets: []}")
    chart_config = models.JSONField(default=dict, help_text="Configuración específica del gráfico")
    
    # Control
    is_active = models.BooleanField(default=True, help_text="Si se muestra en el dashboard")
    order = models.PositiveIntegerField(default=0, help_text="Orden de visualización")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['order', 'created_at']
        verbose_name = "Diagrama"
        verbose_name_plural = "Diagramas"
    
    def __str__(self):
        return f"{self.title} ({self.data_source.name})"
    
    def get_chart_colors(self):
        """Retorna colores para el gráfico basado en tipo"""
        color_schemes = {
            "bar": ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"],
            "line": ["#06b6d4", "#f97316", "#84cc16", "#ec4899", "#6366f1"],
            "pie": ["#f59e0b", "#ef4444", "#10b981", "#3b82f6", "#8b5cf6"],
            "area": ["#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
        }
        return color_schemes.get(self.chart_type, color_schemes["bar"])
    
    def execute_query(self):
        """Ejecuta la consulta SQL y retorna los datos actualizados"""
        try:
            engine = get_engine()
            df = pd.read_sql(self.sql_query, engine)
            
            # Convertir a formato Chart.js
            if len(df.columns) >= 2:
                labels = df.iloc[:, 0].astype(str).tolist()
                values = df.iloc[:, 1].astype(float).tolist()
                
                chart_data = {
                    "labels": labels,
                    "datasets": [{
                        "label": self.title,
                        "data": values,
                        "backgroundColor": self.get_chart_colors()[:len(labels)],
                        "borderColor": self.get_chart_colors()[:len(labels)],
                        "borderWidth": 2
                    }]
                }
                
                # Actualizar datos guardados
                self.chart_data = chart_data
                self.save(update_fields=['chart_data', 'updated_at'])
                
                return chart_data
            else:
                return {"labels": [], "datasets": []}
                
        except Exception as e:
            return {"error": str(e), "labels": [], "datasets": []}
    
    def duplicate(self, new_title=None):
        """Crea una copia del diagrama"""
        new_diagram = Diagrama(
            data_source=self.data_source,
            owner=self.owner,
            title=new_title or f"{self.title} (Copia)",
            description=self.description,
            chart_type=self.chart_type,
            source_type=CHAT,  # Las copias se marcan como del chat
            sql_query=self.sql_query,
            chart_data=self.chart_data.copy(),
            chart_config=self.chart_config.copy(),
            order=self.order + 1
        )
        new_diagram.save()
        return new_diagram

def get_dataset(schema, table):
    engine = get_engine()
    query = f'SELECT * FROM "{schema}"."{table}"'
    df = pd.read_sql(query, engine)
    return df