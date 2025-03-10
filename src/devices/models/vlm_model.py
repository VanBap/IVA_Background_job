from django.db import models

from common.base_model import BaseModel

class VLMModel(BaseModel):

    name = models.CharField(max_length=255, default='', blank=True)
    code_name = models.CharField(max_length=255)
    api_key = models.TextField()
    url = models.CharField(max_length=255, default='', blank=True)

    class Meta:
        managed = False
        db_table = 'vlm_model'