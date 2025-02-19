from django.db import models


# DON'T define objects = models.Manager() here,
# it can override attribute objects of SafeDeleteModel
class BaseSimpleModel(models.Model):
    id = models.BigAutoField(primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)

    class Meta:
        abstract = True


class BaseModel(BaseSimpleModel):
    created_by = models.BigIntegerField(null=True)
    updated_by = models.BigIntegerField(null=True)

    class Meta:
        abstract = True


class BaseModelFK(BaseSimpleModel):
    created_by = models.ForeignKey('user_management.User', null=True, on_delete=models.SET_NULL,
                                   db_column='created_by', related_name='+')
    updated_by = models.ForeignKey('user_management.User', null=True, on_delete=models.SET_NULL,
                                   db_column='updated_by', related_name='+')

    class Meta:
        abstract = True


