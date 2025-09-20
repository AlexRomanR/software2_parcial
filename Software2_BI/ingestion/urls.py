from django.urls import path
from .views import (
    # Vistas existentes (sin cambios)
    upload_dataset_view, list_sources_view, chart_view, user_data_summary_view, 
    delete_source, download_schema, dashboard_list, dashboard_detail, 
    widget_partial, kpi_partial, table_partial, export_csv,
    # Nuevas vistas BI
    data_explorer, create_dashboard, api_chart_data
)

app_name = "ingestion"
urlpatterns = [
    # ============================
    # URLs EXISTENTES (sin cambios)
    # ============================
    path("upload/", upload_dataset_view, name="upload"),
    path("list/", list_sources_view, name="list"),
    path("chart/<int:source_id>/", chart_view, name="chart"),
    path("summary/", user_data_summary_view, name="user_summary"), 
    path("delete-source/<int:source_id>/", delete_source, name="delete_source"),
    path("download-schema/<int:source_id>/", download_schema, name="download_schema"),
    
    # ============================
    # Dashboards interactivos (existentes)
    # ============================
    path("bi/dash/", dashboard_list, name="dashboard_list"),
    path("bi/dash/<int:dashboard_id>/", dashboard_detail, name="dashboard_detail"),
    path("bi/widget/<int:widget_id>/", widget_partial, name="widget_partial"),
    path("bi/kpi/<int:kpi_id>/", kpi_partial, name="kpi_partial"),
    path("bi/table/<int:dashboard_id>/", table_partial, name="table_partial"),
    path("bi/export/<int:dashboard_id>/", export_csv, name="export_csv"),

    # ============================
    # NUEVAS URLs BI (solo agregadas)
    # ============================
    
    # Explorador de datos por archivo
    path("bi/explore/<int:source_id>/", data_explorer, name="data_explorer"),
    
    # Crear dashboard personalizado para un archivo
    path("bi/create-dashboard/<int:source_id>/", create_dashboard, name="create_dashboard"),
    
    # API para datos de gráficos (AJAX/fetch)
    path("api/chart-data/<int:widget_id>/", api_chart_data, name="api_chart_data"),
]