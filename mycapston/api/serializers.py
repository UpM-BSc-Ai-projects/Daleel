from rest_framework import serializers
from base.models import (
    Admin, SecurityStaff, FamilyMember, Person, 
    CameraDetectedPerson, LastSeen, SearchData, ResultsList
)

# ======================================================
# 🧩 Admin Serializer
# ======================================================
class AdminSerializer(serializers.ModelSerializer):
    class Meta:
        model = Admin
        fields = '__all__'

# ======================================================
# 🧩 Security Staff Serializer
# ======================================================
class SecurityStaffSerializer(serializers.ModelSerializer):
    class Meta:
        model = SecurityStaff
        fields = '__all__'

# ======================================================
# 🧩 Family Member Serializer
# ======================================================
class FamilyMemberSerializer(serializers.ModelSerializer):
    class Meta:
        model = FamilyMember
        fields = '__all__'

# ======================================================
# 🧩 CameraDetectedPerson Serializer
# ======================================================
class CameraDetectedPersonSerializer(serializers.ModelSerializer):
    class Meta:
        model = CameraDetectedPerson
        fields = '__all__'

# ======================================================
# 🧩 Person Serializer 
# ======================================================
class PersonSerializer(CameraDetectedPersonSerializer):  # inherits 
    class Meta(CameraDetectedPersonSerializer.Meta):  # also Meta inherits
        model = Person
        fields = [
            'personId', 'firstName', 'lastName', 'age', 'isMale', 
            'idNumber', 'phoneNumber', 'LastLocation', 'description', 
            'video', 'image_1','image_2', 'image_3', 'image_4', 'image_5', 'uploadTime', 'isLost', 'relatTo', 'reportedBy'
        ]

    # The Software solutions to avoid record duplication.
    def validate_phoneNumber(self, value):
        if value:
            instance = getattr(self, 'instance', None)
            qs = Person.objects.filter(phoneNumber=value)
            if instance:
                qs = qs.exclude(pk=instance.pk)  # exclude self in update case

            existing_person = qs.first()
            if existing_person:
                if existing_person.relatTo:
                    relative_name = f"{existing_person.relatTo.firstName} {existing_person.relatTo.lastName}"
                    relative_phone = existing_person.relatTo.phoneNumber
                    msg = f"This phone number is already recorded. It belongs to a person reported by {relative_name} ({relative_phone})."
                elif existing_person.reportedBy:
                    officer_name = f"{existing_person.reportedBy.firstName} {existing_person.reportedBy.lastName}"
                    officer_phone = existing_person.reportedBy.phoneNumber
                    last_location = getattr(existing_person, 'LastLocation', 'unknown location')
                    msg = f"This phone number is already recorded. It belongs to a person reported by officer {officer_name} ({officer_phone}). He is in {last_location}."
                else:
                    msg = "This phone number is already used."
                raise serializers.ValidationError(msg)
        return value


    def validate_idNumber(self, value):
        if value:
            instance = getattr(self, 'instance', None)
            qs = Person.objects.filter(idNumber=value)
            if instance:
                qs = qs.exclude(pk=instance.pk)

            existing_person = qs.first()
            if existing_person:
                if existing_person.relatTo:
                    relative_name = f"{existing_person.relatTo.firstName} {existing_person.relatTo.lastName}"
                    relative_phone = existing_person.relatTo.phoneNumber
                    msg = f"This ID number is already recorded. It belongs to a person reported by {relative_name} ({relative_phone})."
                elif existing_person.reportedBy:
                    officer_name = f"{existing_person.reportedBy.firstName} {existing_person.reportedBy.lastName}"
                    officer_phone = existing_person.reportedBy.phoneNumber
                    last_location = getattr(existing_person, 'LastLocation', 'unknown location')
                    msg = f"This ID number is already recorded. It belongs to a person reported by officer {officer_name} ({officer_phone}). He is in {last_location}."
                else:
                    msg = "This ID number is already used."
                raise serializers.ValidationError(msg)
        return value


    def validate(self, attrs):
        first = attrs.get('firstName')
        last = attrs.get('lastName')
        is_male = attrs.get('isMale')
        age = attrs.get('age')

        instance = getattr(self, 'instance', None)

        # Age range filter for approximate age matching
        age_filter = {}
        if age is not None:
            age_filter = {"age__gte": age - 2, "age__lte": age + 2}

        # Validate exact match (first name + last name + gender + approximate age)
        existing_person_case1 = Person.objects.filter(
            firstName=first,
            lastName=last,
            isMale=is_male,
            **age_filter
        )

        # Validate swapped names (last name + first name + gender + approximate age)
        existing_person_case2 = Person.objects.filter(
            lastName=last,
            isMale=is_male,
            **age_filter
        )

        if instance:
            existing_person_case1 = existing_person_case1.exclude(pk=instance.pk)
            existing_person_case2 = existing_person_case2.exclude(pk=instance.pk)

        existing_person = existing_person_case1.first() or existing_person_case2.first()

        if existing_person:
            if existing_person.relatTo:
                relative_name = f"{existing_person.relatTo.firstName} {existing_person.relatTo.lastName}"
                relative_phone = existing_person.relatTo.phoneNumber
                msg = f"A person with the same name, gender, and close age already exists, reported by {relative_name} ({relative_phone})."
            elif existing_person.reportedBy:
                officer_name = f"{existing_person.reportedBy.firstName} {existing_person.reportedBy.lastName}"
                officer_phone = existing_person.reportedBy.phoneNumber
                last_location = getattr(existing_person, 'LastLocation', 'unknown location')
                msg = f"A person with the same name, gender, and close age already exists, reported by officer {officer_name} ({officer_phone}). He is in {last_location}."
            else:
                msg = "A person with the same name, gender, and close age already exists."
            raise serializers.ValidationError(msg)

        return attrs

# ======================================================
# 🧩 LastSeen Serializer
# ======================================================
class LastSeenSerializer(serializers.ModelSerializer):
    class Meta:
        model = LastSeen
        fields = '__all__'

# ======================================================
# 🧩 Search Data Serializer
# ======================================================
class SearchDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = SearchData
        fields = '__all__'

# ======================================================
# 🧩 Results List Serializer
# ======================================================
class ResultsListSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResultsList
        fields = '__all__'