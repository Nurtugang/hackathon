# Generated by Django 3.2.17 on 2023-04-16 00:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SmartTraffic', '0002_driver_fine'),
    ]

    operations = [
        migrations.AddField(
            model_name='fine',
            name='valid',
            field=models.BooleanField(default=False),
        ),
    ]