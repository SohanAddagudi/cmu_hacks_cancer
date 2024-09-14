from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.predict, name='upload'),
    path('heatmaps/', views.heatmaps_page, name='heatmaps'),
    path('cancinfo/', views.cancinfo, name='cancinfo')
]
