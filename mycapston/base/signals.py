from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import CameraDetectedPerson

"""Signal file to handle automatic actions on model events."""

@receiver(post_save, sender=CameraDetectedPerson)
def run_ai_attributes(sender, instance, created, **kwargs):
    if created:  # only when a new CDP is created
        instance.calculate_ai_attributes()
