from rest_framework.exceptions import ValidationError, NotFound


class CameraNotFound(NotFound):
    default_code = 'camera_not_found'
    default_detail = 'Camera not found'


class CameraGroupNotFound(NotFound):
    default_code = 'camera_group_not_found'
    default_detail = 'Camera group not found'


class CameraNameAlreadyExist(ValidationError):
    default_code = 'camera_name_exist'
    default_detail = 'Name of camera already exists'


class CameraBackgroundInvalid(ValidationError):
    default_code = 'background_camera_invalid'
    default_detail = 'Background camera is invalid'


class ChargingStationNotFound(NotFound):
    default_code = 'charging_station_not_found'
    default_detail = 'Charging station not found'


class EdgeNameAlreadyExist(ValidationError):
    default_code = 'edge_name_exist'
    default_detail = 'Name of edge already exists'


class CameraVideoSegmentNotFound(NotFound):
    default_code = 'camera_video_segment_not_found'
    default_detail = 'Camera video segment not exist'


class CameraVideoSegmentDuplicated(ValidationError):
    default_code = 'camera_video_segment_duplicated'
    default_detail = 'Duplicated video segment'


class VmsNotFound(NotFound):
    default_code = 'vms_not_found'
    default_detail = 'VMS not found'
