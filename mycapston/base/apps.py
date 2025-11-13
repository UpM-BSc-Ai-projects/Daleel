from django.apps import AppConfig


class BaseConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'base'

# I added this to import signals:
    def ready(self): 
        import base.signals  # call signals.py to connect the signals
