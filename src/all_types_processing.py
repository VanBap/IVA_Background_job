import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

import time
import logging
from django.utils.timezone import now

from devices.models.camera import Camera
from devices.models.rule import Rule
from devices.services import camera_alert_service

# === Thread ===
import threading
import concurrent.futures

# === Rule type 0 (Scene Change) ===
from proc.scene_change_detector import process_camera

# === Rule type 1 (Prompt-based) ===
from proc.rule_prompt_proc import process_vlm_rule

logger = logging.getLogger('app')


if __name__ == "__main__":
    print("Which Type do you want to process? ")
    print("0: Scence change detector")
    print("1: Prompt-based detection")
    user_input = int(input("Enter your choice: "))

    print('Loading list of rules')
    rules = Rule.objects.all()

    if user_input == 0:
        while True:

            current_time = now().time()
            print(f"=== Current time: {current_time}")

            # === Thread for Type 0 ===
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for rule in rules:
                    # Chi xu ly neu thoi gian hien tai anm trong [start, end] cu rule
                    if rule.start_time and rule.end_time and rule.start_time <= current_time <= rule.end_time:
                        print(f"=== [Processing] Rule_type {rule.type}")
                        print(f"=== [Processing] Rule_id {rule.id}")
                        print(f"=== [Processing] Checking at time: {current_time}")

                        cameras = rule.cameras.all()
                        # Duyet tung camera trong tung rule
                        for camera in cameras:
                            # === Tao Thread de xu ly kiem tra goc Camera ===
                            future = executor.submit(process_camera, rule, camera)
                            futures.append(future)

            # Cho tat ca cac Thread hoan thanh
            concurrent.futures.wait(futures)
            print(" ")
            print('Process rules finished')
            print(" ")
            # sleep some time
            time.sleep(3)

    # Type == 1
    elif user_input == 1:
        for rule in rules:
            print(f"=== [Processing] Rule_type {rule.type}")
            print(f"=== [Processing] rule_id: {rule.id}")
            cameras = rule.cameras.all()
            for camera in cameras:
                print(f"==== [Processing] camera_id: {camera.id}")
                process_vlm_rule(rule, camera)




