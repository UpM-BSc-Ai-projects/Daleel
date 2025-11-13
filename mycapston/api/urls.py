from django.urls import path, include
from rest_framework.routers import DefaultRouter

from api.views import (
    AdminViewSet,
    SecurityStaffViewSet,
    FamilyMemberViewSet,
    PersonViewSet,
    CameraDetectedPersonViewSet,
    LastSeenViewSet,
    SearchDataViewSet,
    ResultsListViewSet
)

router = DefaultRouter()
router.register(r'admins', AdminViewSet)
router.register(r'security-staff', SecurityStaffViewSet)
router.register(r'family-members', FamilyMemberViewSet)
router.register(r'persons', PersonViewSet)
router.register(r'detections', CameraDetectedPersonViewSet, basename='detection')  # ✅
router.register(r'last-seen', LastSeenViewSet)
router.register(r'searches', SearchDataViewSet)  # ✅
router.register(r'results', ResultsListViewSet)  # ✅

urlpatterns = [
    path('', include(router.urls)),
]

# أو إضافة API root view
urlpatterns += [
    path('api-auth/', include('rest_framework.urls')),
]