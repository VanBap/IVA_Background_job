from django.db import models

from .rule import Rule
from common.base_model import BaseModel

class RuleVersion(BaseModel):

    rule = models.ForeignKey(Rule, on_delete=models.CASCADE, related_name="versions")
    version_number = models.IntegerField()

    name = models.CharField(max_length=255)
    type = models.IntegerField(default=0)
    start_time = models.TimeField(null=True)
    end_time = models.TimeField(null=True)

    class Meta:

        managed = False
        db_table = 'rule_version'

        # UNIQUE rule + version_number
        constraints = [
            models.UniqueConstraint(
                fields=['rule', 'version_number'],
                name='unique_rule_version'
            )
        ]