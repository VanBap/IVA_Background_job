from concurrent.futures import ThreadPoolExecutor
import requests

def fetch_url(url):
    response = requests.get(url)
    return response.status_code

urls = [
    "https://www.google.com",
    "https://www.github.com",
    "https://www.python.org",
    "https://www.stackoverflow.com"
]

# Tạo một thread pool với 4 thread
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(fetch_url, urls))
    print(results)