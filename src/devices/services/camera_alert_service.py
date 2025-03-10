import logging
from time import timezone
from django.db import transaction
from django.utils import timezone
# from common import drf
# from utils.minio_utils import MinioUtil
from ..models import BaseSimpleModel, BaseModel
from ..models import CameraAlert
from ..models import Camera

# === Create a new data in camera_alert if changed is detected ===
logger = logging.getLogger('app')


def model_create(model_class, validated_data, write_db=True):
    camera_name = validated_data.pop('camera_name')

    # set base model field
    if issubclass(model_class, BaseSimpleModel):
        now = timezone.now()
        # model.save will auto set created_at, updated_at. This is a backup.
        validated_data['created_at'] = now

    instance = model_class(**validated_data)
    if getattr(instance, 'custom_clean', None):
        instance.custom_clean()  # noqa

    if write_db:
        instance.save()
    return instance


@transaction.atomic
def create_alert(validated_data):

    # create alert camera
    # validated_data['camera'] = camera
    if validated_data.get('rule_type') == 0:
        validated_data['type'] = "Scence change"
        validated_data['desc'] = "Camera Alert: Goc camera thay doi"

    if validated_data.get('rule_type') == 1:
        validated_data['type'] = "Prompt-based Detection"
        validated_data['desc'] = validated_data.get('desc') + " Answer: Yes"

    validated_data.pop('rule_type')
    return model_create(CameraAlert, validated_data)