# Generated by Django 3.2.18 on 2023-04-16 06:53

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('SmartTraffic', '0008_alter_fine_driver'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fine',
            name='driver',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='SmartTraffic.driver'),
        ),
    ]
