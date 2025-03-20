import base64
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()


from devices.models.rule import Rule

import logging

from proc import scene_change_detector, rule_prompt_detector

logger = logging.getLogger('app')



def process_all_rules():

    print('Loading list of rules')
    rules = Rule.objects.all()
    print('Process rules')

    for rule in rules:
        print(f"=== [Processing] Rule_type {rule.type}")
        print(f"=== [Processing] rule_id: {rule.id}")

        rule_prompt_detector.process_rule_prompt_based()
        scene_change_detector.process_rule_scene_change()
