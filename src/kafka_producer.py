import json
import os
import time

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from kafka import KafkaProducer
from django.utils.timezone import now
from devices.models.rule import Rule

KAFKA_BROKER = 'localhost:10108'
# KAFKA_BROKER = '172.19.0.3:9092'
TOPIC_NAME = 'vannhk_test_030325'

# Khoi tao Kafka Producer
producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER,
                         # api_version=(7, 9, 0),
                         value_serializer=lambda v:json.dumps(v).encode('utf-8'))


def send_camera_task():
    print("[Prodcuer] Loading list of rules ...")
    rules = Rule.objects.all()
    current_time = now().time()

    for rule in rules:
        if rule.start_time and rule.end_time and rule.start_time <= current_time <= rule.end_time:
            print(f"=== rule: {rule.id} HOAT DONG")

            cameras = rule.cameras.all()
            for camera in cameras:
                # Task dang JSON
                task_data = {
                    "rule_id": rule.id,
                    "camera_id": camera.id,
                    "camera_name": camera.name,
                    "version_number": rule.current_version,
                    "camera_url": camera.url,
                    "background_url": camera.background_url,

                }
                key = str(int(camera.id) % 4).encode('utf-8')  # Partition

                producer.send(TOPIC_NAME, value=task_data, key=key)
                print(f"[Producer] Sent task: {task_data}")
        else:
            print(f"=== rule: {rule}: Chua den Thoi gian chi dinh")

    producer.flush()
    print("[Producer] All tasks sent successfully")

if __name__ == "__main__":
    while True:
        send_camera_task()
        time.sleep(3)