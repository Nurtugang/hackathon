from django.db import models
from django.utils import timezone
from django.shortcuts import reverse


class Video(models.Model):
    file = models.FileField(upload_to='files/',null=True)

    def __str__ (self):
        return str(self.file)

VEHICLES = (
    ("car", "car"),
    ("bus", "bus"),
    ("truck", "truck"),
)
class Driver(models.Model):
    name = models.CharField(max_length=100, blank=True)
    surname = models.CharField(max_length=100, blank=True)
    email = models.CharField(max_length=100, blank=True)
    vehicle = models.CharField(max_length=10, choices=VEHICLES, blank=True)
    number = models.CharField(max_length=20, blank=True)
    number_img = models.ImageField(upload_to='num_pics', blank=True)


    def __str__ (self):
        return str(self.number)

class Fine(models.Model):
    driver = models.ForeignKey(Driver, on_delete=models.CASCADE, default=999)
    fine_date = models.DateTimeField(default=timezone.now)
    speed = models.IntegerField()
    valid = models.BooleanField(default=False)
    check = models.BooleanField(default=False)
    accident_img = models.ImageField(upload_to='num_pics', blank=True)
    number_img = models.ImageField(upload_to='num_pics', blank=True)
    def get_absolute_url(self):
        return reverse('concrete_fine', kwargs={'id' : self.id})
    def __str__(self):
        return str(self.speed)