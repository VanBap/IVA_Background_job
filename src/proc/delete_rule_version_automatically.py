import os
import time
import logging
import cv2
from django.db.models import Q

from django.utils.timezone import now
from datetime import timedelta

from devices.models.camera import Camera
from devices.models.rule import Rule
from devices.services import camera_alert_service
from devices.models.rule_version import RuleVersion
from devices.models.camera_alert import CameraAlert



logger = logging.getLogger('app')

N_DAYS = 3
INTERVAL = 10 #seconds

def delete_old_rule_version():

    delete_before = now() - timedelta(days=N_DAYS)
    qs = RuleVersion.objects.filter(created_at__lt = delete_before)
    print(f"============= RuleVersion {qs.values_list('id')} older than {N_DAYS} days. ==============")

    # Xoa cac Rule da qua N
    deleted_count = qs.delete()

    print(f"============= Deleted {deleted_count} old RuleVersion records older than {N_DAYS} days.")

def process():
    while True:
        delete_old_rule_version()
        time.sleep(INTERVAL)