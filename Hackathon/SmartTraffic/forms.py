from django import forms
from .models import *

class VideoForm(forms.ModelForm):
   class Meta:
      model = Video
      fields = ['file']


CHOICES= [
    ('Оштрафовать', 'Оштрафовать'),
    ('Отказать', 'Отказать'),
]
class ValidFineForm(forms.Form):
   number = forms.CharField(label='Госномер машины', max_length=100)
   choice_fine = forms.CharField(label='Что будете делать?', widget=forms.RadioSelect(choices=CHOICES))