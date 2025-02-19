import django
import os


if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    django.setup()

    from proc import delete_rule_version_automatically

    delete_rule_version_automatically.process()



