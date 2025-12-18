import base64
import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.core.mail import EmailMessage
from django.core.serializers.json import DjangoJSONEncoder
from django.db import connection
from django.http import FileResponse, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from sqlalchemy import text

from .models import DataSource, Diagrama, UploadedDataset
from .services import (
    analyze_chart_image,
    generar_consulta_y_grafico,
    generar_diagrama_chat,
    generar_diagramas_automaticos,
    get_dataset,
    get_engine,
    get_schema_info,
    guardar_diagrama,
    import_csv_or_excel,
    import_sql_script,
    obtener_diagramas_por_archivo,
    reduce_schema,
    sanitize_identifier,
)


def api_list_datasets(request, user_id):
    datasets = DataSource.objects.filter(owner_id=user_id).order_by("-created_at")
    data = [
        {
            "id": ds.id,
            "name": ds.name or "Sin nombre",
            "schema": ds.internal_schema or "",
            "table": ds.internal_table or "",
            "created_at": (
                ds.created_at.strftime("%Y-%m-%d %H:%M:%S") if ds.created_at else ""
            ),
        }
        for ds in datasets
    ]
    return JsonResponse({"datasets": data})


def diagramas_por_data_source(request, data_source_id):
    """
    Devuelve todos los diagramas asociados a un DataSource específico.
    """
    try:
        # Filtra por data_source_id y solo los activos
        diagramas = Diagrama.objects.filter(
            data_source_id=data_source_id, is_active=True
        ).order_by("order", "created_at")

        # Serializa la información en JSON
        data = [
            {
                "id": d.id,
                "title": d.title,
                "description": d.description,
                "chart_type": d.chart_type,
                "source_type": d.source_type,
                "sql_query": d.sql_query,
                "chart_data": d.chart_data,
                "order": d.order,
                "created_at": d.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for d in diagramas
        ]

        return JsonResponse({"status": "ok", "diagramas": data}, status=200)

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
