import json
import os
import django
import logging

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()
import cv2

from kafka import KafkaConsumer
from devices.services import camera_alert_service
from proc.extract_thumbnail import extract_thumbnail
from proc.ai_match_scene_images import SceneMatcher

# KAFKA_BROKER = 'kafka:9092'
KAFKA_BROKER = 'localhost:10108'
TOPIC_NAME = 'vannhk_test_050325'

logger = logging.getLogger('app')

# Khoi tao Kafka Consumer

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=KAFKA_BROKER,
    # api_version=(7, 9, 0),
    value_deserializer= lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset= 'earliest',
    group_id='camera_check_group'
)

def check_key_points(task):

    snapshot_dir = '/home/vbd-vanhk-l1-ubuntu/work/'
    os.makedirs(snapshot_dir, exist_ok=True)
    camera_data = task.get("cameras", {})

    snapshot_path = os.path.join(snapshot_dir, f'camera_{camera_data.get("camera_id")}_snapshot.jpg')


    current_image = extract_thumbnail(camera_data.get("camera_url"), snapshot_path)

    img_1 = cv2.imread(task["background_url"])
    img_2 = cv2.imread(current_image)

    matcher = SceneMatcher(visualize=False)
    return matcher.match_scenes(img_1, img_2)

def process_task(task):
    print(f"[Consumer] Processing task: {task}")
    camera_data = task.get("cameras", {})

    if not check_key_points(task):
        alert_data = {
            "rule_id": task["rule_id"],
            "camera_id": camera_data.get("camera_id"),
            'camera_name': camera_data.get("camera_name"),
            "version_number": task["version_number"],
        }
        camera_alert_service.create_alert(alert_data)
        print("[Consumer] Camera alert created! ")
    else:
        print("[Consumer] No change detected.")

if __name__ == "__main__":

    print("[Consumer] Listening for tasks...")
    for message in consumer:
        process_task(message.value)