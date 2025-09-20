# ingestion/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.files.storage import default_storage
from django.urls import reverse
from django.conf import settings
from django.db import connection
from django.core.paginator import Paginator
from django.views.decorators.http import require_POST
from django.http import HttpResponse, FileResponse, JsonResponse
from psycopg2 import sql
import subprocess
import tempfile
import json
import uuid
import csv
import io
import os

from .models import DataSource, UploadedDataset, Dashboard, Widget, KpiCard, DataAnalysis
from .services import (
    import_csv_or_excel,
    import_sql_script,
    sanitize_identifier,
    get_dataset,
    get_schema_info,
)

# ============================
#  Ingesta y gestión de fuentes
# ============================

@login_required
def upload_dataset_view(request):
    """
    Sube CSV/XLSX/SQL, crea un schema único y una tabla interna.
    NUEVO: Genera automáticamente dashboard para el archivo.
    """
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

        try:
            # 3) Importar datos según tipo
            if ext in ("csv", "xlsx"):
                meta = import_csv_or_excel(path, schema, table)
                ds.internal_schema = schema
                ds.internal_table = table
                ds.save(update_fields=["internal_schema", "internal_table"])

                up.rows_ingested = meta.get("rows", 0)
                up.columns = meta.get("columns", [])
                up.save(update_fields=["rows_ingested", "columns"])

            else:  # .sql
                tables, meta_info, main_table = import_sql_script(path, schema)
                ds.internal_schema = schema
                ds.internal_table = main_table or (tables[0] if tables else "")
                ds.save(update_fields=["internal_schema", "internal_table"])

                if main_table and main_table in meta_info:
                    up.rows_ingested = meta_info[main_table].get("rows", 0)
                    up.columns = meta_info[main_table].get("columns", [])
                    up.save(update_fields=["rows_ingested", "columns"])

            # 4) NUEVO: Generar dashboard automático
            auto_dashboard = ds.auto_dashboard
            auto_dashboard.generate_auto_widgets()
            
            messages.success(request, f"Archivo '{ds.name}' importado exitosamente. Dashboard automático creado.")
            return redirect("ingestion:dashboard_detail", dashboard_id=auto_dashboard.id)

        except Exception as e:
            messages.error(request, f"Error al procesar archivo: {str(e)}")
            return render(request, "ingestion/upload.html", {"error": str(e)})

    return render(request, "ingestion/upload.html")


@login_required
def list_sources_view(request):
    """
    Lista los DataSource del usuario con opciones de dashboard.
    """
    sources = DataSource.objects.filter(owner=request.user).order_by("-created_at")
    return render(request, "ingestion/list.html", {"sources": sources})


@login_required
def chart_view(request, source_id):
    """
    Vista simple que carga un dataset y lo envía a un template con gráfico.
    Limita a 5000 filas para no sobrecargar el navegador.
    """
    source = get_object_or_404(DataSource, id=source_id, owner=request.user)
    df = get_dataset(source.internal_schema, source.internal_table)

    # normalizar fecha si existe
    if "fecha" in df.columns:
        try:
            df["fecha"] = df["fecha"].dt.strftime("%Y-%m-%d")
        except Exception:
            df["fecha"] = df["fecha"].astype(str)

    # Limitar por seguridad
    df = df.head(5000)

    chart_data = df.to_dict(orient="records")
    chart_data_json = json.dumps(chart_data, ensure_ascii=False)

    return render(request, "ingestion/chart.html", {
        "source": source,
        "chart_data": chart_data_json
    })


@login_required
def user_data_summary_view(request):
    """
    Resumen de esquemas y tablas internas del usuario.
    """
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
                "tables": tables_info,
                "source": src  # Para acceso a dashboards
            })

    return render(request, "ingestion/user_summary.html", {"all_data": all_data})


@require_POST
@login_required
def delete_source(request, source_id):
    """
    Elimina el schema interno y las filas de tracking para un UploadedDataset.
    NUEVO: También elimina dashboards asociados.
    """
    dataset = get_object_or_404(UploadedDataset, id=source_id, source__owner=request.user)
    schema = dataset.source.internal_schema
    source_name = dataset.source.name

    with connection.cursor() as cursor:
        # 1) Borrar el esquema de forma segura
        if schema:
            cursor.execute(
                sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE;')
                   .format(sql.Identifier(schema))
            )

        # 2) Los dashboards se eliminan automáticamente por CASCADE
        # 3) Borrar registros asociados
        cursor.execute("DELETE FROM ingestion_uploadeddataset WHERE id = %s;", [dataset.id])
        cursor.execute("DELETE FROM ingestion_datasource WHERE id = %s;", [dataset.source.id])

    messages.success(request, f"Archivo '{source_name}' y sus dashboards eliminados correctamente.")
    return redirect("ingestion:list")


@login_required
def download_schema(request, source_id):
    """
    Descarga un dump del schema interno del dataset indicado.
    """
    dataset = get_object_or_404(UploadedDataset, id=source_id, source__owner=request.user)
    schema = dataset.source.internal_schema

    db = settings.DATABASES["default"]
    pg_dump_path = getattr(settings, "PG_DUMP_PATH", r"C:\Program Files\PostgreSQL\17\bin\pg_dump.exe")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".sql")
    tmp_file.close()

    cmd = [
        pg_dump_path,
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

    try:
        subprocess.run(cmd, env=env, check=True)
        return FileResponse(open(tmp_file.name, "rb"), as_attachment=True, filename=f"{schema}.sql")
    finally:
        pass


# ============================
#  Sistema BI - Dashboards por archivo
# ============================

SAFE_LIMIT = 1000

def _bi_apply_conditions(sql_text: str, desde: str | None, hasta: str | None, cond_extra: str | None, date_column: str = "fecha"):
    """
    Inyecta condiciones opcionales en SQL. 
    NUEVO: Parámetro date_column configurable.
    """
    sql_work = sql_text
    params = []

    if "{cond_fecha}" in sql_work:
        cond = ""
        if desde:
            cond += f" AND {date_column} >= %s"
            params.append(desde)
        if hasta:
            cond += f" AND {date_column} <= %s"
            params.append(hasta)
        sql_work = sql_work.replace("{cond_fecha}", cond)

    if "{cond_extra}" in sql_work:
        extra = f" AND {cond_extra}" if cond_extra else ""
        sql_work = sql_work.replace("{cond_extra}", extra)

    if " limit " not in sql_work.lower():
        sql_work += f" LIMIT {SAFE_LIMIT}"

    return sql_work, params


def _bi_run_sql(sql_text: str, params: list):
    """Ejecuta SQL de forma segura y retorna resultados."""
    try:
        with connection.cursor() as cur:
            cur.execute(sql_text, params)
            if cur.description:
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                return rows
            return []
    except Exception as e:
        return [{"error": str(e)}]


@login_required
def dashboard_list(request):
    """
    Lista de dashboards agrupados por archivo de datos.
    """
    # Dashboards automáticos por archivo
    sources_with_dashboards = DataSource.objects.filter(
        owner=request.user,
        dashboards__isnull=False
    ).distinct().order_by("-created_at")
    
    # Dashboards personalizados (sin archivo asociado)
    custom_dashboards = Dashboard.objects.filter(
        owner=request.user,
        data_source__isnull=True
    ).order_by("-created_at")
    
    return render(request, "ingestion/dashboard_list.html", {
        "sources_with_dashboards": sources_with_dashboards,
        "custom_dashboards": custom_dashboards
    })


@login_required
def dashboard_detail(request, dashboard_id: int):
    """
    Vista completa del dashboard con análisis automático.
    """
    dash = get_object_or_404(Dashboard, pk=dashboard_id, owner=request.user)
    
    # Información del archivo asociado
    data_source = dash.data_source
    column_info = data_source.get_column_info() if data_source else {}
    
    context = {
        "dash": dash,
        "data_source": data_source,
        "column_info": column_info,
        "has_date_columns": any('date' in info.get('type', '').lower() or 'timestamp' in info.get('type', '').lower() 
                                for info in column_info.values())
    }
    
    return render(request, "ingestion/dashboard_detail.html", context)


@login_required
def data_explorer(request, source_id: int):
    """
    NUEVA: Explorador de datos para un archivo específico.
    """
    source = get_object_or_404(DataSource, pk=source_id, owner=request.user)
    
    # Obtener muestra de datos
    sample_data = source.upload.get_sample_data(20) if hasattr(source, 'upload') else []
    column_info = source.get_column_info()
    
    context = {
        "source": source,
        "sample_data": sample_data,
        "column_info": column_info,
        "total_rows": source.upload.rows_ingested if hasattr(source, 'upload') else 0
    }
    
    return render(request, "ingestion/data_explorer.html", context)


@login_required
def create_dashboard(request, source_id: int):
    """
    NUEVA: Crear dashboard personalizado para un archivo.
    """
    source = get_object_or_404(DataSource, pk=source_id, owner=request.user)
    
    if request.method == "POST":
        name = request.POST.get("name", f"Dashboard - {source.name}")
        description = request.POST.get("description", "")
        
        dashboard = Dashboard.objects.create(
            name=name,
            description=description,
            owner=request.user,
            data_source=source
        )
        
        messages.success(request, f"Dashboard '{name}' creado exitosamente.")
        return redirect("ingestion:dashboard_detail", dashboard_id=dashboard.id)
    
    return render(request, "ingestion/dashboard_create.html", {"source": source})


@login_required
def widget_partial(request, widget_id: int):
    """
    Parcial HTMX de un widget con colores mejorados.
    """
    w = get_object_or_404(Widget, pk=widget_id, dashboard__owner=request.user)
    desde = request.GET.get("desde")
    hasta = request.GET.get("hasta")
    cond_extra = request.GET.get("cond")

    # Detectar columna de fecha automáticamente si existe
    date_column = "fecha"
    if w.dashboard.data_source:
        column_info = w.dashboard.data_source.get_column_info()
        date_columns = [col for col, info in column_info.items() 
                       if 'date' in info.get('type', '').lower() or 'timestamp' in info.get('type', '').lower()]
        if date_columns:
            date_column = date_columns[0]

    sql_text, params = _bi_apply_conditions(w.sql.strip(), desde, hasta, cond_extra, date_column)
    data = _bi_run_sql(sql_text, params)

    # Manejar errores
    if data and "error" in data[0]:
        return render(request, "ingestion/_widget_error.html", {"w": w, "error": data[0]["error"]})

    labels = [str(d.get("label", "")) for d in data]
    values = [float(d.get("value", 0)) if d.get("value") is not None else 0 for d in data]
    series = [str(d.get("series", "")) for d in data] if data and "series" in data[0] else []

    # Configurar datasets con colores
    datasets = []
    colors = w.get_chart_colors()
    
    if series:
        series_names = sorted(set(series))
        for i, s in enumerate(series_names):
            dsrows = [d for d in data if str(d.get("series", "")) == s]
            labs = [str(d.get("label", "")) for d in dsrows]
            vals = [float(d.get("value", 0)) if d.get("value") is not None else 0 for d in dsrows]
            
            datasets.append({
                "label": s,
                "data": vals,
                "backgroundColor": colors[i % len(colors)] + "80",  # 50% transparencia
                "borderColor": colors[i % len(colors)],
                "borderWidth": 2
            })
        if series_names:
            labels = [str(d.get("label", "")) for d in data if str(d.get("series", "")) == series_names[0]]
    else:
        datasets = [{
            "label": w.title,
            "data": values,
            "backgroundColor": [color + "80" for color in colors[:len(values)]],
            "borderColor": colors[:len(values)],
            "borderWidth": 2
        }]

    return render(request, "ingestion/_widget_partial.html", {
        "w": w, 
        "labels": labels, 
        "datasets": datasets,
        "chart_colors": colors
    })


@login_required
def kpi_partial(request, kpi_id: int):
    """
    Parcial HTMX de una tarjeta KPI mejorada.
    """
    k = get_object_or_404(KpiCard, pk=kpi_id, dashboard__owner=request.user)
    desde = request.GET.get("desde")
    hasta = request.GET.get("hasta")
    cond_extra = request.GET.get("cond")

    # Detectar columna de fecha
    date_column = "fecha"
    if k.dashboard.data_source:
        column_info = k.dashboard.data_source.get_column_info()
        date_columns = [col for col, info in column_info.items() 
                       if 'date' in info.get('type', '').lower() or 'timestamp' in info.get('type', '').lower()]
        if date_columns:
            date_column = date_columns[0]

    sql_text, params = _bi_apply_conditions(k.sql.strip(), desde, hasta, cond_extra, date_column)
    data = _bi_run_sql(sql_text, params)
    
    value = None
    if data and not data[0].get("error"):
        raw_value = data[0].get("value")
        if raw_value is not None:
            # Formatear números grandes
            if isinstance(raw_value, (int, float)):
                if raw_value >= 1000000:
                    value = f"{raw_value/1000000:.1f}M"
                elif raw_value >= 1000:
                    value = f"{raw_value/1000:.1f}K"
                else:
                    value = f"{raw_value:,.0f}"
            else:
                value = str(raw_value)

    return render(request, "ingestion/_kpi_partial.html", {"k": k, "value": value})


@login_required
def table_partial(request, dashboard_id: int):
    """
    Tabla detalle dinámica basada en el archivo del dashboard.
    """
    dash = get_object_or_404(Dashboard, pk=dashboard_id, owner=request.user)
    desde = request.GET.get("desde")
    hasta = request.GET.get("hasta")
    cond_extra = request.GET.get("cond")

    if not dash.data_source or not dash.data_source.internal_schema:
        return render(request, "ingestion/_table_partial.html", {
            "dash": dash, 
            "page_obj": None,
            "error": "No hay datos asociados a este dashboard"
        })

    # Construir SQL dinámico basado en las columnas del archivo
    schema = dash.data_source.internal_schema
    table = dash.data_source.internal_table
    column_info = dash.data_source.get_column_info()
    
    # Limitar columnas mostradas (máximo 6)
    display_columns = list(column_info.keys())[:6]
    columns_sql = ", ".join([f'"{col}"' for col in display_columns])
    
    # Detectar columna de fecha para ordenamiento
    date_columns = [col for col, info in column_info.items() 
                   if 'date' in info.get('type', '').lower() or 'timestamp' in info.get('type', '').lower()]
    order_by = f'"{date_columns[0]}" DESC' if date_columns else f'"{display_columns[0]}"'
    
    base_sql = f"""
    SELECT {columns_sql}
    FROM "{schema}"."{table}"
    WHERE 1=1 {{cond_fecha}} {{cond_extra}}
    ORDER BY {order_by}
    """

    date_column = date_columns[0] if date_columns else "fecha"
    sql_text, params = _bi_apply_conditions(base_sql, desde, hasta, cond_extra, date_column)
    
    # Remover LIMIT para paginación
    sql_text = sql_text.replace(f" LIMIT {SAFE_LIMIT}", "")
    
    rows = _bi_run_sql(sql_text, params)

    page = int(request.GET.get("page", 1))
    paginator = Paginator(rows, 25)
    page_obj = paginator.get_page(page)

    return render(request, "ingestion/_table_partial.html", {
        "dash": dash, 
        "page_obj": page_obj,
        "display_columns": display_columns
    })


@login_required
def export_csv(request, dashboard_id: int):
    """
    Exporta datos del dashboard como CSV.
    """
    dash = get_object_or_404(Dashboard, pk=dashboard_id, owner=request.user)
    desde = request.GET.get("desde")
    hasta = request.GET.get("hasta")
    cond_extra = request.GET.get("cond")

    if not dash.data_source:
        return HttpResponse("No hay datos para exportar", content_type="text/plain")

    schema = dash.data_source.internal_schema
    table = dash.data_source.internal_table
    
    base_sql = f'SELECT * FROM "{schema}"."{table}" WHERE 1=1 {{cond_fecha}} {{cond_extra}}'
    
    column_info = dash.data_source.get_column_info()
    date_columns = [col for col, info in column_info.items() 
                   if 'date' in info.get('type', '').lower() or 'timestamp' in info.get('type', '').lower()]
    date_column = date_columns[0] if date_columns else "fecha"
    
    sql_text, params = _bi_apply_conditions(base_sql, desde, hasta, cond_extra, date_column)
    sql_text = sql_text.replace(f" LIMIT {SAFE_LIMIT}", "")  # Sin límite para export
    
    rows = _bi_run_sql(sql_text, params)

    buf = io.StringIO()
    fieldnames = rows[0].keys() if rows else []
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

    resp = HttpResponse(buf.getvalue(), content_type="text/csv; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="{dash.data_source.name}_export.csv"'
    return resp


# ============================
#  API para gráficos dinámicos
# ============================

@login_required
def api_chart_data(request, widget_id: int):
    """
    NUEVA: API JSON para datos de gráficos (para actualizaciones dinámicas).
    """
    w = get_object_or_404(Widget, pk=widget_id, dashboard__owner=request.user)
    desde = request.GET.get("desde")
    hasta = request.GET.get("hasta")
    cond_extra = request.GET.get("cond")

    sql_text, params = _bi_apply_conditions(w.sql.strip(), desde, hasta, cond_extra)
    data = _bi_run_sql(sql_text, params)

    return JsonResponse({
        "labels": [str(d.get("label", "")) for d in data],
        "values": [float(d.get("value", 0)) if d.get("value") is not None else 0 for d in data],
        "chart_type": w.chart_type,
        "title": w.title,
        "colors": w.get_chart_colors()
    })