from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action

from base.models import (
    Admin,
    SecurityStaff,
    FamilyMember,
    Person,
    CameraDetectedPerson,
    LastSeen,
    SearchData,
    ResultsList
)

from .serializers import (
    AdminSerializer,
    SecurityStaffSerializer,
    FamilyMemberSerializer,
    PersonSerializer,
    CameraDetectedPersonSerializer,
    LastSeenSerializer,
    SearchDataSerializer,
    ResultsListSerializer
)

# ======================================================
# 🧩 Base ViewSet
# ======================================================
class BaseViewSet(viewsets.ModelViewSet):
    # a base viewset to inherit common behaviors
    
    @action(detail=False, methods=['get'])
    def count(self, request):
        # to get total count of records
        count = self.get_queryset().count()
        return Response({'count': count})
    
    @action(detail=False, methods=['get'])
    def recent(self, request):
        # to get recent records, default 10
        limit = int(request.query_params.get('limit', 10))
        recent = self.get_queryset().order_by('-pk')[:limit]
        serializer = self.get_serializer(recent, many=True)
        return Response(serializer.data)

# ======================================================
# 🧩 Admin ViewSet
# ======================================================
class AdminViewSet(BaseViewSet):
    queryset = Admin.objects.all()
    serializer_class = AdminSerializer

# ======================================================
# 🧩 Security Staff ViewSet
# ======================================================
class SecurityStaffViewSet(BaseViewSet):
    queryset = SecurityStaff.objects.all()
    serializer_class = SecurityStaffSerializer

# ======================================================
# 🧩 Family Member ViewSet
# ======================================================
class FamilyMemberViewSet(BaseViewSet):
    queryset = FamilyMember.objects.all()
    serializer_class = FamilyMemberSerializer

# ======================================================
# 🧩 CameraDetectedPerson ViewSet (super class for Person)
# ======================================================
class CameraDetectedPersonViewSet(BaseViewSet):
    queryset = CameraDetectedPerson.objects.all()
    serializer_class = CameraDetectedPersonSerializer
    
    @action(detail=False, methods=['get']) # detail=False so no pk is needed, foer all records
    def potentially_lost(self, request):
        # to get all potentially lost persons
        potentially_lost = CameraDetectedPerson.objects.filter(potentiallyLost=True)
        serializer = self.get_serializer(potentially_lost, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def elderly_persons(self, request):
        # to get all elderly persons
        elderly = CameraDetectedPerson.objects.filter(isElderly=True)
        serializer = self.get_serializer(elderly, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def disabled_persons(self, request):
        # to get all disabled persons
        disabled = CameraDetectedPerson.objects.filter(isDisabled=True)
        serializer = self.get_serializer(disabled, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post']) # to update the potentially lost, elderly, disabled fields
    def update_cdp(self, request, pk=None):
        # to update CameraDetectedPerson details
        cdp = self.get_object()
        serializer = self.get_serializer(cdp, data=request.data, partial=True)
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
# ======================================================    need to think
    @action(detail=True, methods=['post'])
    def add_last_seen(self, request, pk=None):
        # to add a last seen location
        cdp = self.get_object()
        location = request.data.get('location')
        coordinates = request.data.get('coordinates')
        time = request.data.get('time')  # optional
        
        if not location or not coordinates:
            return Response(
                {'error': 'Location and coordinates are required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        cdp.addLastSeen(location, coordinates, time)
        return Response({'message': 'Last seen location added successfully'})

# ======================================================
# 🧩 Person ViewSet (subclass of CameraDetectedPersonViewSet)
# ======================================================
class PersonViewSet(CameraDetectedPersonViewSet):  # inherits from CameraDetectedPersonViewSet
    queryset = Person.objects.all()
    serializer_class = PersonSerializer
    
    @action(detail=False, methods=['get'])
    def lost_persons(self, request):
        # to get all lost persons
        lost_persons = Person.objects.filter(isLost=True)
        serializer = self.get_serializer(lost_persons, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def found_persons(self, request):
        # to get all found persons
        found_persons = Person.objects.filter(isLost=False)
        serializer = self.get_serializer(found_persons, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def quick_search(self, request):
        # to search persons by basic attributes
        first_name = request.query_params.get('firstName', '')
        last_name = request.query_params.get('lastName', '')
        age = request.query_params.get('age', '')
        id_number = request.query_params.get('idNumber', '')
        is_male = request.query_params.get('isMale', '')
        
        queryset = Person.objects.all()
        
        if first_name:
            queryset = queryset.filter(firstName__icontains=first_name)
        if last_name:
            queryset = queryset.filter(lastName__icontains=last_name)
        if age:
            queryset = queryset.filter(age=age)
        if id_number:
            queryset = queryset.filter(idNumber__icontains=id_number)
        if is_male.lower() == 'true':
            queryset = queryset.filter(isMale=True)
        elif is_male.lower() == 'false':
            queryset = queryset.filter(isMale=False)
        
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def update_status(self, request, pk=None):
        """تحديث حالة الشخص (مفقود/تم العثور عليه)"""
        person = self.get_object()
        is_lost = request.data.get('isLost')
        
        if is_lost is not None:
            if not is_lost:  # إذا تم العثور عليه
                person.updateStatus()  # يستخدم الدالة الموجودة في الموديل
                return Response({'message': 'Person found and record deleted successfully'})
            else:
                person.isLost = is_lost
                person.save()
                return Response({'message': 'Status updated successfully'})
        
        return Response({'error': 'isLost field is required'}, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['post'])
    def process_ai(self, request, pk=None):
        """تشغيل الذكاء الاصطناعي على الشخص"""
        person = self.get_object()
        person.doAI()
        return Response({'message': 'AI processing completed successfully'})
    
    @action(detail=False, methods=['get'])
    def search_by_family(self, request):
        """البحث عن الأشخاص المرتبطين بعائلة معينة"""
        family_member_id = request.query_params.get('family_member_id')
        
        if family_member_id:
            persons = Person.objects.filter(relatTo_id=family_member_id)
            serializer = self.get_serializer(persons, many=True)
            return Response(serializer.data)
        
        return Response({'error': 'family_member_id is required'}, status=status.HTTP_400_BAD_REQUEST)

# ======================================================
# 🧩 LastSeen ViewSet
# ======================================================
class LastSeenViewSet(BaseViewSet):
    queryset = LastSeen.objects.all()
    serializer_class = LastSeenSerializer
    
    @action(detail=False, methods=['get'])
    def recent_sightings(self, request):
        """الحصول على المشاهدات الحديثة"""
        from django.utils import timezone
        from datetime import timedelta
        
        hours = int(request.query_params.get('hours', 24))
        time_threshold = timezone.now() - timedelta(hours=hours)
        
        recent_sightings = LastSeen.objects.filter(time__gte=time_threshold)
        serializer = self.get_serializer(recent_sightings, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def by_location(self, request):
        """البحث عن المشاهدات حسب الموقع"""
        location = request.query_params.get('location', '')
        
        if location:
            sightings = LastSeen.objects.filter(location__icontains=location)
            serializer = self.get_serializer(sightings, many=True)
            return Response(serializer.data)
        
        return Response({'error': 'location parameter is required'}, status=status.HTTP_400_BAD_REQUEST)

# ======================================================
# 🧩 Search Data ViewSet
# ======================================================
class SearchDataViewSet(BaseViewSet):
    queryset = SearchData.objects.all()
    serializer_class = SearchDataSerializer
    
    @action(detail=True, methods=['post'])
    def update_status(self, request, pk=None):
        """تحديث حالة البحث"""
        search_data = self.get_object()
        is_found = request.data.get('isFound')
        
        if is_found is not None:
            search_data.updateStatus(is_found)
            serializer = self.get_serializer(search_data)
            return Response(serializer.data)
        
        return Response({'error': 'isFound field is required'}, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['get'])
    def active_searches(self, request):
        """الحصول على البحوث النشطة"""
        active_searches = SearchData.objects.filter(isProcessing=True)
        serializer = self.get_serializer(active_searches, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def completed_searches(self, request):
        """الحصول على البحوث المكتملة"""
        completed_searches = SearchData.objects.filter(isProcessing=False)
        serializer = self.get_serializer(completed_searches, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def successful_searches(self, request):
        """الحصول على البحوث الناجحة"""
        successful_searches = SearchData.objects.filter(isFound=True)
        serializer = self.get_serializer(successful_searches, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['delete'])
    def cancel_search(self, request, pk=None):
        """إلغاء البحث"""
        search_data = self.get_object()
        try:
            search_data.deleteSearch()
            return Response({'message': 'Search deleted successfully'})
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )

# ======================================================
# 🧩 Results List ViewSet
# ======================================================
class ResultsListViewSet(BaseViewSet):
    queryset = ResultsList.objects.all()
    serializer_class = ResultsListSerializer
    
    @action(detail=False, methods=['get'])
    def accepted_results(self, request):
        """الحصول على النتائج المقبولة"""
        accepted_results = ResultsList.objects.filter(isAccepted=True)
        serializer = self.get_serializer(accepted_results, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def rejected_results(self, request):
        """الحصول على النتائج المرفوضة"""
        rejected_results = ResultsList.objects.filter(isAccepted=False)
        serializer = self.get_serializer(rejected_results, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def by_search(self, request):
        """الحصول على نتائج بحث معين"""
        search_id = request.query_params.get('search_id')
        
        if search_id:
            results = ResultsList.objects.filter(searchID=search_id)
            serializer = self.get_serializer(results, many=True)
            return Response(serializer.data)
        
        return Response({'error': 'search_id parameter is required'}, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['get'])
    def by_person(self, request):
        """الحصول على نتائج لشخص معين"""
        person_id = request.query_params.get('person_id')
        
        if person_id:
            results = ResultsList.objects.filter(cameraDetectedPersonId=person_id)
            serializer = self.get_serializer(results, many=True)
            return Response(serializer.data)
        
        return Response({'error': 'person_id parameter is required'}, status=status.HTTP_400_BAD_REQUEST)