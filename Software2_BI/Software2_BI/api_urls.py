from django.urls import include, path

urlpatterns = [
    path("", include("accounts.api_urls")),  # Rutas API de accounts
    path("", include("ingestion.urls")),  # Rutas API de ingestion
]
