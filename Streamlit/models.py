from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, JSON, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base, engine
import random
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_URL = "http://localhost:6333"

# note: we are using SQLAlchemy ORM to define our models which will be mapped to SQLlite database tables

class Admin(Base):
    __tablename__ = "admins"
    __table_args__ = {'extend_existing': True}
    adminId = Column(Integer, primary_key=True, index=True) # index=True for faster search
    name = Column(String(200))
    email = Column(String(200))
    passwordHash = Column(String(255))
    phoneNumber = Column(String(20))

class SecurityStaff(Base):
    __tablename__ = "security_staff"
    __table_args__ = {'extend_existing': True}

    staffId = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    email = Column(String(200))
    passwordHash = Column(String(255))
    phoneNumber = Column(String(20))

    # Relationships
    # this and relationship came together (powrerful facilatator of SQLAlchemy)
    # to link staff to admin e.g staff = get_security_staff_by_id(db, 1), print(staff.admin.name)
    admin_id = Column(Integer, ForeignKey("admins.adminId", ondelete='SET NULL'), nullable=True) # if admin is deleted, set to NULL
    # Relationship to Admin (string target; resolved via SQLAlchemy registry)
    admin = relationship("Admin", backref="security_staffs")

class FamilyMember(Base):
    __tablename__ = "family_members"
    __table_args__ = {'extend_existing': True}

    familyMemberId = Column(Integer, primary_key=True, index=True)
    firstName = Column(String(100))
    lastName = Column(String(100))
    idNumber = Column(String(25)) # national ID number for security verification
    phoneNumber = Column(String(20))
    relation = Column(String(50)) # e.g Father, Mother, Sibling

class CameraDetectedPerson(Base):
    __tablename__ = "camera_detected_persons"
    __table_args__ = {'extend_existing': True}

    # Use a Python-side default generator for the primary key so inserts from
    # CRUD helpers (which don't explicitly set the ID) don't violate NOT NULL.
    cameraDetectedPersonId = Column(
        String(10),
        primary_key=True,
        autoincrement=False,
        default=lambda: str(random.randint(1000000000, 9999999999)),
    )
    potentiallyLost = Column(Boolean, default=False)
    isElderly = Column(Boolean, default=False)
    isDisabled = Column(Boolean, default=False)

    def calculate_ai_attributes(self):
        """
        Placeholder / default AI attributes calculation.
        In the current design, actual AI flags are computed in the
        embedding worker and synced from Qdrant. This method exists
        only so that legacy calls (e.g. from CRUD helpers) don't fail.
        """
        # No-op implementation – keep current values as-is.
        # You can later replace this with real logic if needed.
        return self

class Person(Base):
    __tablename__ = "persons" # indirectly inherits from CameraDetectedPerson
    __table_args__ = {'extend_existing': True}

    personId = Column(Integer, primary_key=True, index=True)
    
    # relationship to CameraDetectedPerson
    # if CameraDetectedPerson is deleted, Person remains (SET NULL)
    cameraDetectedPersonId = Column(String(10), ForeignKey('camera_detected_persons.cameraDetectedPersonId', ondelete='SET NULL'), nullable=True)    
    camera_detected_person = relationship("CameraDetectedPerson", backref="persons")  # inverse relationship: CameraDetectedPerson.persons
    
    firstName = Column(String(100))
    lastName = Column(String(100))
    age = Column(Integer, nullable=True)
    isMale = Column(Boolean, default=True)
    idNumber = Column(String(25))
    phoneNumber = Column(String(20))
    LastLocation = Column(String(255), default="Unknown")
    description = Column(Text)
    video = Column(String(500), nullable=True)
    image_1 = Column(String(500), nullable=True)
    image_2 = Column(String(500), nullable=True)
    image_3 = Column(String(500), nullable=True)
    image_4 = Column(String(500), nullable=True)
    image_5 = Column(String(500), nullable=True)
    uploadTime = Column(DateTime, default=func.now())
    isLost = Column(Boolean, default=True)
    
    # Relationships
    # if FamilyMember is deleted, Person is deleted (CASCADE)
    relatTo_id = Column(Integer, ForeignKey("family_members.familyMemberId", ondelete='CASCADE'), nullable=True)
    relatTo = relationship("FamilyMember", backref="persons")  # inverse relationship: FamilyMember.persons
    
    # if SecurityStaff is deleted, Person remains (SET NULL)
    reportedBy_id = Column(Integer, ForeignKey("security_staff.staffId", ondelete='SET NULL'), nullable=True) 
    reportedBy = relationship("SecurityStaff", backref="reported_persons")  # inverse relationship: SecurityStaff.reported_persons

class LastSeen(Base):
    __tablename__ = "last_seen"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    location = Column(String(255), default="Camira XXX")
    time = Column(DateTime, default=func.now())
    coordinates = Column(JSON, default=list)
    
    # if CameraDetectedPerson is deleted, LastSeen is deleted (CASCADE)
    CDPid = Column(String(10), ForeignKey("camera_detected_persons.cameraDetectedPersonId", ondelete='CASCADE'))
    cdp = relationship("CameraDetectedPerson", backref="last_seen_records")  # inverse relationship: CameraDetectedPerson.last_seen_records

class SearchData(Base):
    __tablename__ = "search_data"
    __table_args__ = {'extend_existing': True}

    searchID = Column(Integer, primary_key=True, index=True)
    requestTime = Column(DateTime, default=func.now())
    isProcessing = Column(Boolean, default=True)
    isFound = Column(Boolean, default=False)
    
    # if SecurityStaff is deleted, SearchData remains (SET NULL)
    requestedBy_id = Column(Integer, ForeignKey("security_staff.staffId", ondelete='SET NULL'))
    requestedBy = relationship("SecurityStaff", backref="search_requests")  # inverse relationship: SecurityStaff.search_requests

class ResultsList(Base):
    __tablename__ = "results_list"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    isAccepted = Column(Boolean, default=False)

    # Relationships
    # if SearchData is deleted, ResultsList is deleted (CASCADE)
    searchID = Column(Integer, ForeignKey("search_data.searchID", ondelete='CASCADE'))
    searchData = relationship("SearchData", backref="results")  # inverse relationship: SearchData.results
    
    # if CameraDetectedPerson is deleted, ResultsList is deleted (CASCADE)
    cameraDetectedPersonId = Column(String(10), ForeignKey("camera_detected_persons.cameraDetectedPersonId", ondelete='CASCADE'))
    camera_detected_person = relationship("CameraDetectedPerson", backref="search_results")  # inverse relationship: CameraDetectedPerson.search_results

def create_tables():
    Base.metadata.create_all(bind=engine)

def initialize_collections():
    client = QdrantClient(QDRANT_URL)
    
    try:
        client.create_collection(
        collection_name='CLIP_embeddings',
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
    except:
        print("CLIP_embeddings collection exists")
    
    try:
        client.create_collection(
        collection_name='stream',
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
    except:
        print("stream collection exists")
    
