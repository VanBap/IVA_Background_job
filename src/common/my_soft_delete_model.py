import datetime
from django.db import models


def get_timestamp_ms():
    return int(datetime.datetime.now().timestamp() * 1000)


class SoftDeleteQuerySet(models.query.QuerySet):
    def delete(self):
        return self.update(deleted_at=get_timestamp_ms())

    def hard_delete(self):
        return super().delete()


class SoftDeleteManager(models.Manager):
    def get_queryset(self):
        return SoftDeleteQuerySet(self.model, using=self._db).filter(deleted_at=0)


class DeletedQuerySet(models.query.QuerySet):
    def restore(self, *args, **kwargs):
        qs = self.filter(*args, **kwargs)
        qs.update(deleted_at=0)


class DeletedManager(models.Manager):
    def get_queryset(self):
        return DeletedQuerySet(self.model, using=self._db).exclude(deleted_at=0)


class GlobalManager(models.Manager):
    pass


class MySoftDeleteModel(models.Model):
    deleted_at = models.BigIntegerField(default=0)

    objects = SoftDeleteManager()
    deleted_objects = DeletedManager()
    global_objects = GlobalManager()

    class Meta:
        abstract = True

    def delete(self, cascade=None, *args, **kwargs):
        self.deleted_at = get_timestamp_ms()
        self.save(update_fields=['deleted_at'])

    def restore(self):
        self.deleted_at = 0
        self.save(update_fields=['deleted_at'])

    def hard_delete(self, *args, **kwargs):
        super().delete(*args, **kwargs)
