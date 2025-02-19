import os

INSTALLED_APPS = ['devices']

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'HOST': os.getenv('MYSQL_DB_HOST'),
        'PORT': os.getenv('MYSQL_DB_PORT'),
        'NAME': os.getenv('MYSQL_DB_NAME'),
        'USER': os.getenv('MYSQL_DB_USER'),
        'PASSWORD': os.getenv('MYSQL_DB_PASSWORD'),
        'TIME_ZONE': 'Asia/Saigon',
    }
}

SECRET_KEY = 'lkjsdafklkdgfklsdmlfkmdsklfmkdssdkalmsa'

USE_TZ = True
TIME_ZONE = 'Asia/Saigon'  # or UTC
