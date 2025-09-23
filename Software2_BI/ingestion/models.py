from django.db import models
from django.contrib.auth import get_user_model
from fernet_fields import EncryptedTextField, EncryptedCharField

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

    
def get_dataset(schema, table):
    engine = get_engine()
    query = f'SELECT * FROM "{schema}"."{table}"'
    df = pd.read_sql(query, engine)
    return df