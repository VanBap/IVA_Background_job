import itertools
import os
import time
import logging
import types
from collections.abc import Callable

import cv2
import queue

from django.utils.timezone import now

from devices.models.camera import Camera
from devices.models.rule import Rule
from devices.services import camera_alert_service

from proc.extract_thumbnail import extract_thumbnail
from proc.ai_match_scene_images import SceneMatcher



# === Thread ===
import threading
import concurrent.futures
from concurrent.futures import _base
import itertools
import queue
import threading
import types
import weakref
import os

# === Implement MyThreadPool using Queue ===
import queue
import sys
from collections.abc import Callable, Iterable, Mapping, Set as AbstractSet
from threading import Lock, Semaphore, Thread
from typing import Any, Generic, TypeVar, overload
from typing_extensions import TypeVarTuple, Unpack
from weakref import ref

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

def process_camera(rule, camera):
    # Xu ly tung camera trong 1 rule.
    thread_name = threading.current_thread().name
    print(f" === [{thread_name}] === Processing rule: {rule.id} - camera: {camera.id}")

    # Neu goc Cam thay doi => Tao data trong bang Camera Alert
    if check_key_points(camera) is False:
        try:
            data = {
                'rule_id': rule.id,
                'camera_id': camera.id,
                'camera_name': camera.name,
                'version_number': rule.current_version,
            }
            print(f"=== data: {data}")

            camera_alert_service.create_alert(data)  # tao camera trong bang camera alert
            print('Camera alert service created')
        except AttributeError as e:
            print(f"Loi tao camera alert: {e}")
    else:
        print(f"Goc camera {camera.id} khong thay doi")


# === Queue manages works ===

NUM_WORKERS = 5  # So luong worker thread

class MyThreadPool():
    def __init__(self, num_threads=5):
        self.task_queue = queue.Queue()
        self.num_threads = num_threads
        self.threads = []
        self.create_threads()

    def worker(self, thread_name):
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            try:
                func, args = task
                print(f"[{thread_name}] Processing task: {args}")
                func(*args)  # Extract parameter -> Run function

            finally:
                self.task_queue.task_done()

    def create_threads(self):
    # === Initialize background Thread ===
        for i in range(self.num_threads):
            thread_name = f"Thread-{i + 1}"
            thread = threading.Thread(target=self.worker, args =(thread_name,), name=thread_name ,daemon=True) # args =(thread_name,): add parameter into worker
            thread_name = threading.current_thread().name
            print(f" === [{thread_name}]")

            thread.start()
            self.threads.append(thread)

    def submit(self, fn, args): # add Task to Queue
        self.task_queue.put((fn, args))


    def wait_for_completion(self):
        # Wait for task is done.
        # While it's done, thread is still running in background
        self.task_queue.join()

    def stop_threads(self):
        # Stop all "background" threads by adding None to Queue

        for _ in self.threads:
            self.task_queue.put(None)
        for thread in self.threads:
            thread.join()


def worker(rule, camera):
    # If camera angle changes => Create alert
    if not check_key_points(camera):
        try:
            data = {
                'rule_id': rule.id,
                'camera_id': camera.id,
                'camera_name': camera.name,
                'version_number': rule.current_version,
            }

            camera_alert_service.create_alert(data)
            print('Camera alert service created')
        except AttributeError as e:
            print(f"Loi tao camera alert: {e}")

    else:
        print(f"Goc camera {camera.id} khong thay doi")



def process():
    while True:

        print('Loading list of rules')
        rules = Rule.objects.all()
        print('Process rules')

        current_time = now().time()
        print(f"=== Current time: {current_time}")

        # === Thread Queue ===
        mythreadpool = MyThreadPool(NUM_WORKERS)

        for rule in rules:
            # Chi xu ly neu thoi gian hien tai anm trong [start, end] cu rule
            if rule.start_time and rule.end_time and rule.start_time <= current_time <= rule.end_time:
                print(f"=== rule: {rule.id} HOAT DONG")
                cameras = rule.cameras.all()
                # Duyet tung camera trong tung rule
                for camera in cameras:

                    mythreadpool.submit(worker, (rule, camera))

            else:
                print(f"=== rule: {rule}: Chua den Thoi gian chi dinh")

        # Cho tat ca cac Thread hoan thanh
        mythreadpool.wait_for_completion()
        mythreadpool.stop_threads()

        print(" ")
        print('Process rules finished')
        print(" ")

    # sleep some time
    time.sleep(3)


