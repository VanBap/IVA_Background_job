# Generated by Django 5.1.4 on 2025-02-03 04:32

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('devices', '0005_rule_rulecamera_alter_cameraalert_camera'),
    ]

    operations = [
        migrations.AddField(
            model_name='cameraalert',
            name='rule',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.DO_NOTHING, related_name='alerts', to='devices.rule'),
        ),
    ]
