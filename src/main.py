import django
import os


if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    django.setup()

    from proc import scene_change_detector

    scene_change_detector.process()
