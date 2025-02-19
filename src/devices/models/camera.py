from typing import Optional
from enum import IntEnum

from strenum import StrEnum
from django.db import models

from .base import BaseModel

from common.my_soft_delete_model import MySoftDeleteModel


from .camera_group import CameraGroup
from utils import exceptions as module_exception

STREAM_FRAME_WIDTH = 1280
STREAM_FRAME_HEIGHT = 720
DEFAULT_FIRST_SAMPLE_TIME = 0
DEFAULT_INTERVAL_SAMPLE_TIME = 20


class ConnStatus(IntEnum):
    CONNECT: int = 1
    NOT_CONNECT: int = 2
    PASSWORD_INVALID: int = 3


class AIFeature(StrEnum):
    FACE = "face"
    HUMAN = "human"
    VEHICLE = "vehicle"
    FIRE = "fire"
    PET = "pet"


class Protocol(IntEnum):
    HIKVISION: int = 1
    ONVIF: int = 2

    @staticmethod
    def get_protocol(protocol: int) -> Optional[str]:
        if protocol == Protocol.HIKVISION:
            return "HIKVISION"
        if protocol == Protocol.ONVIF:
            return "ONVIF"
        return None


class ConnectionType(IntEnum):
    RTSP: int = 1
    NVR: int = 2


class AddConnStatus:
    NOT_ADD_YET: int = 1
    WAIT_TO_ADD: int = 2
    RUNNING: int = 3
    ERROR: int = 4
    WARNING: int = 5


class CameraType(IntEnum):
    LIVE: int = 1
    NON_LIVE: int = 2


class CameraCategory(IntEnum):
    IP_CAMERA: int = 1
    SOC_VIN_BIGDATA: int = 2
    SOC_MK_VISION: int = 3


class Camera(BaseModel, MySoftDeleteModel):
    name = models.CharField(max_length=255)

    desc = models.CharField(max_length=1024, blank=True, default='')

    url = models.CharField(max_length=255, default='', blank=True)

    background_url = models.CharField(max_length=255, default='', blank=True)
    background_width = models.IntegerField(null=True, default=0)
    background_height = models.IntegerField(null=True, default=0)

    # Camera group
    group = models.ForeignKey(CameraGroup, on_delete=models.DO_NOTHING, null=True, related_name='cameras')
    conn_status = models.IntegerField(null=True, default=ConnStatus.NOT_CONNECT)

    class Meta:
        db_table = 'camera'
        managed = False




