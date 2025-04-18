# Generated by Django 5.1.4 on 2025-01-09 09:21

import devices.models.camera
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Camera',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
                ('created_by', models.BigIntegerField(null=True)),
                ('updated_by', models.BigIntegerField(null=True)),
                ('deleted_at', models.BigIntegerField(default=0)),
                ('name', models.CharField(max_length=255)),
                ('desc', models.CharField(blank=True, default='', max_length=1024)),
                ('url', models.CharField(blank=True, default='', max_length=255)),
                ('background_url', models.CharField(blank=True, default='', max_length=255)),
                ('background_width', models.IntegerField(default=0, null=True)),
                ('background_height', models.IntegerField(default=0, null=True)),
                ('conn_status', models.IntegerField(default=devices.models.camera.ConnStatus['NOT_CONNECT'], null=True)),
            ],
            options={
                'db_table': 'camera',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='CameraGroup',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
                ('name', models.CharField(max_length=255)),
                ('active', models.SmallIntegerField(default=1)),
                ('desc', models.CharField(default='', max_length=255)),
            ],
            options={
                'db_table': 'camera_group',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='CameraAlert',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
                ('name', models.CharField(max_length=255)),
                ('desc', models.CharField(default='', max_length=255)),
                ('captured_at', models.DateTimeField()),
            ],
            options={
                'db_table': 'camera_alert',
                'managed': True,
            },
        ),
    ]
