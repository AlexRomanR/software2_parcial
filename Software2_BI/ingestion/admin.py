from django.contrib import admin
from .models import Dashboard, Widget, KpiCard, SavedQuery

@admin.register(Dashboard)
class DashboardAdmin(admin.ModelAdmin):
    list_display = ("id","name","owner","created_at")
    search_fields = ("name","owner__username")

@admin.register(Widget)
class WidgetAdmin(admin.ModelAdmin):
    list_display = ("id","dashboard","title","chart_type","order")
    list_filter = ("chart_type",)
    search_fields = ("title","sql")

@admin.register(KpiCard)
class KpiCardAdmin(admin.ModelAdmin):
    list_display = ("id","dashboard","title","order","suffix")

@admin.register(SavedQuery)
class SavedQueryAdmin(admin.ModelAdmin):
    list_display = ("id","title","owner","is_approved","created_at")
    list_filter = ("is_approved",)
    search_fields = ("title","sql","owner__username")
