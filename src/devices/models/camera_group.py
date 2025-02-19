from django.db import models

from .base import BaseSimpleModel


class CameraGroup(BaseSimpleModel):
    name = models.CharField(max_length=255)
    active = models.SmallIntegerField(default=1)
    desc = models.CharField(max_length= 255, default='')

    class Meta:
        db_table = 'camera_group'
        managed = False
