import os
import time
import logging
import cv2
from django.forms.models import model_to_dict
from django.utils.timezone import now

from devices.models.camera import Camera
from devices.models.rule import Rule
from devices.services import camera_alert_service

from proc.extract_thumbnail import extract_thumbnail
from proc.ai_match_scene_images import SceneMatcher
from devices.serializers.camera_alert_serializer import CameraAlertFilterSerializer

logger = logging.getLogger('app')

# Capture current image
def snapshot_image_from_camera(camera_id, url):
    snapshot_dir = '/home/vbd-vanhk-l1-ubuntu/work/'
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_path = os.path.join(snapshot_dir, f'camera_{camera_id}_snapshot.jpg')

    snapshot_image = extract_thumbnail(url, snapshot_path)

    return snapshot_image

# Detect Camera change
def check_key_points(camera):
    background_url = camera.background_url
    current_image = snapshot_image_from_camera(camera.id, camera.url)
    img_1 = cv2.imread(background_url)
    img_2 = cv2.imread(current_image)

    matcher = SceneMatcher(visualize=False)
    is_matched = matcher.match_scenes(img_1, img_2)
    return is_matched



def process():
    while True:

        print('Loading list of rules')
        rules = Rule.objects.all()
        print('Process rules')

        current_time = now().time()
        print(f"=== Current time: {current_time}")
        for rule in rules:
            # Chi xu ly neu thoi gian hien tai anm trong [start, end] cu rule
            if rule.start_time and rule.end_time and rule.start_time <= current_time <= rule.end_time:
                cameras = rule.cameras.all()
                # Duyet tung camera trong tung rule
                for camera in cameras:
                    print(f"=== rule: {rule} - camera: {camera}")

                    # Neu goc Cam thay doi => Tao data trong bang Camera Alert
                    if check_key_points(camera) is False:
                        try:

                            data = {
                                'rule_id': rule.id,
                                'camera_id': camera.id,
                                'camera_name': camera.name,
                                'version_number': rule.current_version,
                            }
                            print("=== data: ", data)

                            # serializer = CameraAlertFilterSerializer(data=data)
                            # serializer.is_valid(raise_exception=True)
                            # validated_data = serializer.validated_data

                            camera_alert_service.create_alert(data) # tao camera trong bang camera alert
                            print('Camera alert service created')
                        except AttributeError as e:
                            print("Loi tao camera alert: ", e)
                    else:
                        print(f"Goc camera {camera.id} khong thay doi")

                # sleep some time
                time.sleep(3)

            else:
                print(f"=== rule: {rule}: Chua den Thoi gian chi dinh")


