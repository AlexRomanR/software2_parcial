from django.urls import path

from .views import config, panel, predict, train

urlpatterns = [
    path("<str:schema>/<str:table>/", panel, name="panel"),
    path("<str:schema>/<str:table>/config", config, name="config"),
    path("<str:schema>/<str:table>/train", train, name="train"),
    path("<str:schema>/<str:table>/predict", predict, name="predict"),
]
