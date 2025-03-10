from django.db import models

from common.base_model import BaseModel

class Prompt(BaseModel):

    content = models.TextField()
    system = models.TextField(blank=True)

    class Meta:
        managed = False
        db_table = 'prompt'