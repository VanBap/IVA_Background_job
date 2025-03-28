# Generated by Django 5.1.4 on 2025-02-14 04:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('devices', '0008_cameraalert_version_number'),
    ]

    operations = [
        migrations.CreateModel(
            name='RuleVersion',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
                ('created_by', models.BigIntegerField(null=True)),
                ('updated_by', models.BigIntegerField(null=True)),
                ('version_number', models.IntegerField()),
                ('name', models.CharField(max_length=255)),
                ('type', models.IntegerField(default=0)),
                ('start_time', models.TimeField(null=True)),
                ('end_time', models.TimeField(null=True)),
            ],
            options={
                'db_table': 'rule_version',
                'managed': False,
            },
        ),
    ]
