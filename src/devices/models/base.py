from django.db import models


class BaseSimpleModel(models.Model):
    id = models.BigAutoField(primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)

    objects = models.Manager()

    class Meta:
        abstract = True

class BaseModel(BaseSimpleModel):
    created_by = models.BigIntegerField(null=True)
    updated_by = models.BigIntegerField(null=True)

    class Meta:
        abstract = True
