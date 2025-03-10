from django.db import models

from common.base_model import BaseSimpleModel
from .camera import Camera

class Rule(BaseSimpleModel):
    TYPE_CHOICES = [
        (0, "Scene Change"),
        (1, "Prompt-based Detection")
    ]

    name = models.CharField(max_length=255)
    type = models.IntegerField(choices=TYPE_CHOICES, default=0)
    start_time = models.TimeField(null=True)
    end_time = models.TimeField(null=True)

    # many to many
    cameras = models.ManyToManyField(Camera, through='RuleCamera', through_fields=('rule', 'camera'),
                                     related_name="rules")

    # version
    current_version = models.IntegerField(default=1)

    # VLM model
    vlm_model = models.ForeignKey("VLMModel", on_delete=models.DO_NOTHING, null = True, blank = True)

    # prompt
    # Prompt dung cho loai rule TYPE = 1
    # Quan hệ 1-1 với Prompt
    prompt = models.ForeignKey("Prompt", on_delete=models.DO_NOTHING, null=True, blank=True)

    class Meta:
        managed = False
        db_table = 'rule'