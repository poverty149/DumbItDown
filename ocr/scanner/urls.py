from django.urls import path
from django.views.generic.base import TemplateView

from . import views

urlpatterns = [
    path("", views.upload_image, name="index"),
    # path("",TemplateView.as_view(template_name="base.html"),name="base")
]