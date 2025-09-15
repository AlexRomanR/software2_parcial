from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.urls import reverse
from .models import DataSource, UploadedDataset
from .services import import_csv_or_excel, import_sql_script, sanitize_identifier, get_dataset, get_schema_info
import json
import uuid

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