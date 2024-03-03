from django.urls import path 
from .views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', index, name='index'),
    path('home', home, name='home'),
    path('fine', fine, name='fine'),
    path('concrete_fine/<int:id>/', concrete_fine, name='concrete_fine'), 

]