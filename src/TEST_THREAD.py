import threading
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

def log_task():
    while True:
        logging.debug("Logging from daemon thread")
        time.sleep(2)

daemon_thread = threading.Thread(target=log_task)
daemon_thread.daemon = False
daemon_thread.start()

for i in range(5):
    logging.debug("Logging from main thread")
    time.sleep(1)

print("Main thread is exiting...")