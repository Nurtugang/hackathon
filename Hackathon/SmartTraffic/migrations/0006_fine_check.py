# Generated by Django 3.2.17 on 2023-04-16 01:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SmartTraffic', '0005_fine_accident_img'),
    ]

    operations = [
        migrations.AddField(
            model_name='fine',
            name='check',
            field=models.BooleanField(default=False),
        ),
    ]