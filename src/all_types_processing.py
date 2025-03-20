import os
import django
import argparse
import json
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

import logging

from proc import scene_change_detector, rule_prompt_detector, all_rules_detector

logger = logging.getLogger('app')

# === GET TYPE OPTION 3 (ENV PARA) ===
# RULE_TYPE = os.getenv('RULE_TYPE')


if __name__ == "__main__":

    # === GET TYPE OPTION 1 (JSON File) ===
    with open("/home/vbd-vanhk-l1-ubuntu/PycharmProjects/PythonProject/src/config.json") as f:
        config = json.load(f)

    RULE_TYPE = config["rule_type"]["type_1"]
    # ========================================

    # # === GET TYPE OPTION 2 (command line argument) ===
    # parser = argparse.ArgumentParser(description='Pass in Rule type')
    # parser.add_argument("--rule_type", type=str, required=True)
    # args = parser.parse_args()
    #
    # RULE_TYPE = args.rule_type
    # # ========================================

    if RULE_TYPE == '0':
        print("=== [Processing] RULE TYPE 0 ONLY")
        scene_change_detector.process_rule_scene_change()

    # Type == 1
    elif RULE_TYPE == '1':
        print("=== [Processing] RULE TYPE 1 ONLY")
        rule_prompt_detector.process_rule_prompt_based()

    elif RULE_TYPE == 'all':
        print("=== [Processing] RULE ALL TYPES")
        all_rules_detector.process_all_rules()


