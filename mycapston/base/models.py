from django.db import models
import random # delete later

# my notes: blank=True is used to make the testing data entry easier
# ======================================================
# 🧩 Admin
# ======================================================
class Admin(models.Model):
    adminId = models.AutoField(primary_key=True)
    name = models.CharField(max_length=200, blank=True)
    email = models.CharField(max_length=200, blank=True)
    passwordHash = models.CharField(max_length=255, blank=True)
    phoneNumber = models.CharField(max_length=20, blank=True)

# ======================================================
# 🧩 Security Staff
# ======================================================
class SecurityStaff(models.Model):
    staffId = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, blank=True)
    email = models.CharField(max_length=200, blank=True)
    passwordHash = models.CharField(max_length=255, blank=True)
    phoneNumber = models.CharField(max_length=20, blank=True)
    admin = models.ForeignKey(
        Admin,
        related_name="admin",
        on_delete=models.SET_NULL,
        null=True,
    )

# ======================================================
# 🧩 Family Member
# ======================================================
class FamilyMember(models.Model):
    familyMemberId = models.AutoField(primary_key=True)
    firstName = models.CharField(max_length=100, blank=True)
    lastName = models.CharField(max_length=100, blank=True)
    idNumber = models.CharField(max_length=25, blank=True)
    phoneNumber = models.CharField(max_length=20, blank=True)
    relation = models.CharField(max_length=50, blank=True)

# ======================================================
# 🧩 CameraDetectedPerson (CDP)
# ======================================================
class CameraDetectedPerson(models.Model):
    cameraDetectedPersonId = models.AutoField(primary_key=True)
    embedding = models.JSONField(default=list)  # To store the embedding array [70]
    potentiallyLost = models.BooleanField(default=False)
    isElderly = models.BooleanField(default=False)
    isDisabled = models.BooleanField(default=False)  

    ai_processed = models.BooleanField(default=False) # Tracks if our script has run
    def calculate_ai_attributes(self): # in signals.py this method is called after creation utomatically
        self.embedding = [round(random.random(), 4) for _ in range(70)]
        self.potentiallyLost = random.choice([True, False])
        self.isElderly = random.choice([True, False])
        self.isDisabled = random.choice([True, False])   
        self.save() # save the updated attributes to the database

    def addLastSeen(self, location=None, coordinates=None, time=None):
        from django.utils import timezone
        # a father method to create and link LastSeen to this CDP
        LastSeen.objects.create(
            CDPid=self,
            location=location if location is not None else "Camira XXX",
            coordinates=coordinates if coordinates is not None else [],
            time=time if time is not None else timezone.now()
        )


# ======================================================
# 🧩 Person (Lost Person)
# ======================================================
class Person(CameraDetectedPerson): # inherit from CameraDetectedPerson to get embedding etc.
    personId = models.AutoField(primary_key=True)
    firstName = models.CharField(max_length=100, blank=True)
    lastName = models.CharField(max_length=100, blank=True)
    age = models.PositiveIntegerField(blank=True, null=True)
    isMale = models.BooleanField(default=True)
    idNumber = models.CharField(max_length=25, blank=True)
    phoneNumber = models.CharField(max_length=20, blank=True)
    LastLocation = models.CharField(max_length=255, blank=True, default= "Unknown") # last known location by family or real location by Sequrty staff
    description = models.TextField(blank=True) # no max length
    video = models.FileField(upload_to='person_videos/',null=True, blank=True) # the general file field
    image_1 = models.ImageField(upload_to='person_images/', null=True, blank=True) # image field inherits from file field
    image_2 = models.ImageField(upload_to='person_images/', null=True, blank=True)
    image_3 = models.ImageField(upload_to='person_images/', null=True, blank=True)
    image_4 = models.ImageField(upload_to='person_images/', null=True, blank=True)
    image_5 = models.ImageField(upload_to='person_images/', null=True, blank=True)
    uploadTime = models.DateTimeField(auto_now_add=True) # not editable
    isLost = models.BooleanField(default=True) # False when found
    relatTo  = models.ForeignKey(FamilyMember,
        related_name="relatTo",
        on_delete=models.CASCADE,
        null=True, # so security staff can add a lost person without family member info
        blank=True
    ) # example family_member = p.relatTo.all() returns the family member of person p   
    reportedBy = models.ForeignKey( # reported by which security staff, it must be here not in family member
        SecurityStaff,
        related_name="servicedBy",
        on_delete=models.SET_NULL, # no problem if the staff is deleted, all can see the request history
        null=True,
        blank=True
    )

    def updateStatus(self):
        self.isLost = False
        self.save()
        self.deletePerson() # if found, delete the person record
        
    def deletePerson(self):
        try:
            self.relatTo.delete()
        except Exception:
            # there is no related FamilyMember to delete
            pass
        self.delete()

    def doAI(self):
        super().calculate_ai_attributes()

# ======================================================
# 🧩 LastSeen
# ======================================================
class LastSeen(models.Model): # after detection, creat a LastSeen and link to CDP by ForeignKey is easier than comosition
    location = models.CharField(max_length=255, default="Camira XXX")
    time = models.CharField(max_length=255, blank=True, null=True)
    coordinates = models.JSONField(default=list)  # defult = []
    CDPid = models.ForeignKey(
        CameraDetectedPerson,
        related_name="CDPid",
        on_delete=models.CASCADE,
        null=False # to ensure the composition relationship
    )

# ======================================================
# 🧩 Search Data
# ======================================================
class SearchData(models.Model):
    searchID = models.AutoField(primary_key=True)
    requestTime = models.DateTimeField(auto_now_add=True) # not editable, if needed do another search request
    isProcessing = models.BooleanField(default=True) # False when search is done
    isFound = models.BooleanField(default=False) # why shall I search if he is found?
    requestedBy = models.ForeignKey(
        SecurityStaff,
        related_name="requestedBy",
        on_delete=models.SET_NULL, # no problem if the staff is deleted, all can see the request history
        null=True,
    )
    #personId2 = models.ForeignKey(
    #    Person,
    #    related_name="personId2",
    #    on_delete=models.CASCADE,
    #    null=False,
    #)

    def updateStatus(self, isFound: bool):
        self.isFound = isFound
        self.isProcessing = False
        self.save()

    def deleteSearch(self):
        if self.isProcessing:
            raise Exception("Cannot delete a processing search.")
        self.delete()

    #potinsalPersons = models.ManyToManyField([1],[2],[3])
    #for i in Persons:
    #    def addResultsList(self, isAccepted: bool):
    #        # a father method to create and link LastSeen to this CDP
    #        ResultsList.objects.create(
    #            searchID=self,  # as object django will use it to set the ForeignKey
    #            cameraDetectedPersonId=self,
    #            isAccepted=isAccepted
    #        )

# ======================================================
# 🧩 Results List
# ======================================================
class ResultsList(models.Model): # because one search can have multiple accepted/rejected CDPs
    searchID = models.ForeignKey(SearchData, on_delete=models.CASCADE)
    cameraDetectedPersonId = models.ForeignKey(CameraDetectedPerson, on_delete=models.CASCADE)
    isAccepted = models.BooleanField(default=False) # False means rejected

    class Meta:
        unique_together = ('searchID', 'cameraDetectedPersonId', 'isAccepted')
