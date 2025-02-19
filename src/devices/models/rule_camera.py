from django.db import models

from common.base_model import BaseModel
from ..models.camera import Camera
from ..models.rule import Rule

class RuleCamera(BaseModel):
    rule = models.ForeignKey(Rule, on_delete=models.DO_NOTHING, related_name='camera_configs')
    camera = models.ForeignKey(Camera, on_delete=models.DO_NOTHING, related_name='rule_configs')

    class Meta:
        managed = False
        db_table = 'rule_camera'
