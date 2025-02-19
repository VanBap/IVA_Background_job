from concurrent.futures import ThreadPoolExecutor
import time

def worker(name):
    print(f"{name} is working")
    time.sleep(2)
    print(f"{name} is done")

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(worker, f"Thread-{i+1}") for i in range(5)]

print("Exiting Main Thread")