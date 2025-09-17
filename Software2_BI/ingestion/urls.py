from django.urls import path
from .views import upload_dataset_view, list_sources_view, chart_view,user_data_summary_view, delete_source, download_schema

app_name = "ingestion"
urlpatterns = [
    path("upload/", upload_dataset_view, name="upload"),
    path("list/", list_sources_view, name="list"),
    path("chart/<int:source_id>/", chart_view, name="chart"),
    path("summary/", user_data_summary_view, name="user_summary"), 
    path("delete-source/<int:source_id>/", delete_source, name="delete_source"),
    path("download-schema/<int:source_id>/", download_schema, name="download_schema"),
]
