from django.db import models

from common.base_model import BaseSimpleModel
from .camera import Camera

class Rule(BaseSimpleModel):
    name = models.CharField(max_length=255)
    type = models.IntegerField(default=0)
    start_time = models.TimeField(null=True)
    end_time = models.TimeField(null=True)

    # many to many
    cameras = models.ManyToManyField(Camera, through='RuleCamera', through_fields=('rule', 'camera'),
                                     related_name="rules")

    # version
    current_version = models.IntegerField(default=1)

    class Meta:
        managed = False
        db_table = 'rule'