import django
import os


if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    django.setup()

    from proc import scene_change_detector, rule_prompt_detector
    from proc import implement_threadpool_queue
    import TEST_THREAD

    # Kafka
    # import kafka_producer
    # import kafka_consumer
    #
    # kafka_producer.run()

    rule_prompt_detector.process_rule_prompt_based()
