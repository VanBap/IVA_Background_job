import django
import os


if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    django.setup()

    from proc import scene_change_detector
    from proc import implement_threadpool_queue
    implement_threadpool_queue.process()
