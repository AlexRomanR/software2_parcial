from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.urls import reverse
from .models import DataSource, UploadedDataset
from .services import import_csv_or_excel, import_sql_script, sanitize_identifier, get_dataset, get_schema_info, generar_consulta_y_grafico
import json
import uuid
from django.views.decorators.http import require_POST
from django.shortcuts import get_object_or_404
from django.db import connection
from django.http import HttpResponse  
from django.conf import settings
import subprocess
import tempfile
from django.http import FileResponse
import os
from sqlalchemy import text
from .services import get_engine
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.serializers.json import DjangoJSONEncoder
from .services import get_schema_info, generar_consulta_y_grafico, reduce_schema
@login_required
def upload_dataset_view(request):
    if request.method == "POST" and request.FILES.get("file"):
        f = request.FILES["file"]
        ext = f.name.split(".")[-1].lower()
        if ext not in ("csv", "xlsx", "sql"):
            return render(request, "ingestion/upload.html", {"error": "Solo CSV, XLSX o SQL"})

        # 1) Crear registro DataSource
        ds = DataSource.objects.create(
            name=request.POST.get("name", f.name),
            kind=DataSource.FILE,
            owner=request.user,
        )
        up = UploadedDataset.objects.create(source=ds, file=f, file_type=ext)

        # 2) Definir schema y tabla únicos
        schema = sanitize_identifier(f"user_{request.user.id}_file_{uuid.uuid4().hex[:8]}")

        table = sanitize_identifier(f"ds_{ds.id}")


        path = up.file.path

        # 3) Importar datos según tipo
        if ext in ("csv", "xlsx"):
            meta = import_csv_or_excel(path, schema, table)
            ds.internal_schema = schema
            ds.internal_table = table
            ds.save(update_fields=["internal_schema", "internal_table"])
            up.rows_ingested = meta["rows"]
            up.columns = meta["columns"]
            up.save(update_fields=["rows_ingested", "columns"])

        else:  # .sql
            tables, meta_info, main_table = import_sql_script(path, schema)
            ds.internal_schema = schema
            ds.internal_table = main_table or (tables[0] if tables else "")
            ds.save(update_fields=["internal_schema", "internal_table"])

            if main_table:
                up.rows_ingested = meta_info[main_table]["rows"]
                up.columns = meta_info[main_table]["columns"]
                up.save(update_fields=["rows_ingested", "columns"])

        return redirect("ingestion:list")

    return render(request, "ingestion/upload.html")

@login_required
def list_sources_view(request):
    sources = DataSource.objects.filter(owner=request.user).order_by("-created_at")
    return render(request, "ingestion/list.html", {"sources": sources})



def chart_view(request, source_id):
    source = DataSource.objects.get(id=source_id)
    df = get_dataset(source.internal_schema, source.internal_table)

    if "fecha" in df.columns:
        df["fecha"] = df["fecha"].astype(str)


    # Convertir a lista de diccionarios
    chart_data = df.to_dict(orient="records")

    # Convertir a JSON válido
    chart_data_json = json.dumps(chart_data, ensure_ascii=False)

    return render(request, "ingestion/chart.html", {
        "source": source,
        "chart_data": chart_data_json
    })



def user_data_summary_view(request):
    user = request.user
    sources = DataSource.objects.filter(owner=user).order_by("-created_at")

    all_data = []
    for src in sources:
        schema = src.internal_schema
        if schema:
            tables_info = get_schema_info(schema)
            all_data.append({
                "file": src.name,
                "schema": schema,
                "tables": tables_info
            })

    return render(request, "ingestion/user_summary.html", {"all_data": all_data})


@require_POST
def delete_source(request, source_id):
    # Obtenemos el dataset y su DataSource relacionado
    dataset = get_object_or_404(UploadedDataset, id=source_id)
    schema = dataset.source.internal_schema
    table = dataset.source.internal_table

    with connection.cursor() as cursor:
        # 1) Borrar el esquema y su contenido
        cursor.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE;')

        # 2) Borrar registros de las tablas asociadas a este dataset
        cursor.execute("DELETE FROM ingestion_uploadeddataset WHERE id = %s;", [dataset.id])
        cursor.execute("DELETE FROM ingestion_datasource WHERE id = %s;", [dataset.source.id])
        cursor.execute("DELETE FROM ingestion_externalconnection WHERE source_id = %s;", [dataset.source.id])

    return redirect("dashboard")  # Ajusta a la vista a la que quieras volver


def download_schema(request, source_id):
    dataset = get_object_or_404(UploadedDataset, id=source_id)
    schema = dataset.source.internal_schema  # el schema del dataset

    db = settings.DATABASES["default"]

    # Crear archivo temporal
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".sql")
    tmp_file.close()

    # Comando pg_dump
    cmd = [
        "C:/Program Files/PostgreSQL/17/bin/pg_dump.exe",
        "-h", db["HOST"],
        "-p", str(db["PORT"]),
        "-U", db["USER"],
        "-d", db["NAME"],
        "--no-owner",
        "--schema", schema,
        "-f", tmp_file.name
    ]

    env = os.environ.copy()
    env["PGPASSWORD"] = db["PASSWORD"]

    # Ejecutar pg_dump
    subprocess.run(cmd, env=env, check=True)

    # Devolver archivo como descarga
    return FileResponse(open(tmp_file.name, "rb"), as_attachment=True, filename=f"{schema}.sql")



def prueba_view(request):
    sql = None
    grafico = None
    datos = []

    # 📌 Traer nombre de archivo + schema del usuario
    user = request.user
    sources = DataSource.objects.filter(owner=user).order_by("-created_at")

    archivos = []
    for src in sources:
        if src.internal_schema:
            archivos.append({
                "file": src.name,       # nombre del archivo
                "schema": src.internal_schema  # schema real en DB
            })

    if request.method == "POST":
        schema_seleccionado = request.POST.get("schema")  # el schema real
        pregunta = request.POST.get("pregunta")

        # ✅ Obtener esquema de tablas para ese schema real
        esquema = get_schema_info(schema_seleccionado)

        # ✅ Generar SQL y tipo gráfico con Gemini
        sql, grafico = generar_consulta_y_grafico(esquema, pregunta)

        # ✅ Ejecutar SQL si existe
        if sql:
            with connection.cursor() as cursor:
                cursor.execute(f"SET search_path TO {schema_seleccionado}")
                cursor.execute(sql)
                columnas = [col[0] for col in cursor.description]
                datos = [dict(zip(columnas, fila)) for fila in cursor.fetchall()]

    return render(request, "ingestion/prueba.html", {
        "archivos": archivos,   # 📌 Enviamos la lista al template
        "sql": sql,
        "grafico": grafico,
        "datos": datos
    })



def obtener_esquema_bd(schema_name):
    """Obtiene tablas y columnas del esquema seleccionado"""
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = %s
            ORDER BY table_name, ordinal_position
        """, [schema_name])
        filas = cursor.fetchall()

    esquema = {}
    for tabla, col in filas:
        esquema.setdefault(tabla, []).append(col)

    return esquema



def dashboard(request):
    engine = get_engine()
    with engine.begin() as conn:
        res = conn.execute(text("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name LIKE 'user_%'
            ORDER BY schema_name
        """))
        schemas = [r[0] for r in res.fetchall()]

    return render(request, "dashboard.html", {"schemas": schemas})

@csrf_exempt
def prueba_chat_view(request):
    if request.method != "POST":
        user = request.user
        sources = DataSource.objects.filter(owner=user).order_by("-created_at")
        archivos = [{"file": s.name, "schema": s.internal_schema} for s in sources if s.internal_schema]
        from django.shortcuts import render
        return render(request, "ingestion/prueba.html", {"archivos": archivos})

    body = json.loads(request.body or "{}")
    schema = body.get("schema")
    mensaje = body.get("mensaje", "").strip()

    esquema_full = get_schema_info(schema)         # trae columns/rows/preview...
    esquema_reducido = reduce_schema(esquema_full) # SOLO {tabla: [columnas]}

    sql, grafico, respuesta = generar_consulta_y_grafico(esquema_reducido, mensaje)

    columns, datos, error = [], [], None
    if sql and schema:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'SET search_path TO "{schema}"')
                cursor.execute(sql)
                columns = [c[0] for c in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                datos = [list(r) for r in rows]
        except Exception as e:
            error = str(e)

    return JsonResponse(
        {
            "respuesta": respuesta or "",
            "sql": sql or "",
            "grafico": grafico or "bar",
            "columns": columns,
            "datos": datos,
            "error": error,
        },
        encoder=DjangoJSONEncoder,
        json_dumps_params={"ensure_ascii": False}
    )