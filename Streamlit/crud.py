from sqlalchemy.orm import Session
from models import *
from person_validators import validate_person_creation
import random
from datetime import datetime, timedelta

# ======================================================
# 🧩 Admin CRUD
# ======================================================
def create_admin(db: Session, name: str, email: str, password_hash: str, phone_number: str):
    admin = Admin(name=name, email=email, passwordHash=password_hash, phoneNumber=phone_number)
    db.add(admin)
    db.commit()
    db.refresh(admin)
    return admin

def get_admins(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Admin).offset(skip).limit(limit).all()

def get_admin_by_id(db: Session, admin_id: int):
    return db.query(Admin).filter(Admin.adminId == admin_id).first()

def get_admin_with_staff(db: Session, admin_id: int):
    # query to get admin with related security staffs
    admin = db.query(Admin).filter(Admin.adminId == admin_id).first()
    if admin:
        return admin, admin.security_staffs  #  using backref relationship
    return None, []

def update_admin(db: Session, admin_id: int, **kwargs):
    admin = db.query(Admin).filter(Admin.adminId == admin_id).first()
    if admin:
        for key, value in kwargs.items():
            setattr(admin, key, value)
        db.commit()
        db.refresh(admin)
    return admin

def delete_admin(db: Session, admin_id: int):
    admin = db.query(Admin).filter(Admin.adminId == admin_id).first()
    if admin:
        db.delete(admin)
        db.commit()
    return admin

def get_admins_count(db: Session):
    return db.query(Admin).count()

def get_recent_admins(db: Session, limit: int = 10):
    return db.query(Admin).order_by(Admin.adminId.desc()).limit(limit).all()

# ======================================================
# 🧩 Security Staff CRUD
# ======================================================
def create_security_staff(db: Session, name: str, email: str, password_hash: str, phone_number: str, admin_id: int = None):
    staff = SecurityStaff(name=name, email=email, passwordHash=password_hash, phoneNumber=phone_number, admin_id=admin_id)
    db.add(staff)
    db.commit()
    db.refresh(staff)
    return staff

def get_security_staffs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(SecurityStaff).offset(skip).limit(limit).all()

def get_security_staff_by_id(db: Session, staff_id: int):
    return db.query(SecurityStaff).filter(SecurityStaff.staffId == staff_id).first()

def get_security_staff_with_relations(db: Session, staff_id: int):
    # query to get security staff with related reported persons and search requests
    staff = db.query(SecurityStaff).filter(SecurityStaff.staffId == staff_id).first()
    if staff:
        return staff, staff.reported_persons, staff.search_requests  # using backref relationships
    return None, [], []

def update_security_staff(db: Session, staff_id: int, **kwargs):
    staff = db.query(SecurityStaff).filter(SecurityStaff.staffId == staff_id).first()
    if staff:
        for key, value in kwargs.items():
            setattr(staff, key, value)
        db.commit()
        db.refresh(staff)
    return staff

def delete_security_staff(db: Session, staff_id: int):
    staff = db.query(SecurityStaff).filter(SecurityStaff.staffId == staff_id).first()
    if staff:
        db.delete(staff)
        db.commit()
    return staff

def get_security_staff_count(db: Session):
    return db.query(SecurityStaff).count()

def get_recent_security_staff(db: Session, limit: int = 10):
    return db.query(SecurityStaff).order_by(SecurityStaff.staffId.desc()).limit(limit).all()

# ======================================================
# 🧩 Family Member CRUD
# ======================================================
def create_family_member(db: Session, first_name: str, last_name: str, id_number: str, phone_number: str, relation: str):
    member = FamilyMember(firstName=first_name, lastName=last_name, idNumber=id_number, phoneNumber=phone_number, relation=relation)
    db.add(member)
    db.commit()
    db.refresh(member)
    return member

def get_family_members(db: Session, skip: int = 0, limit: int = 100):
    return db.query(FamilyMember).offset(skip).limit(limit).all()

def get_family_member_by_id(db: Session, member_id: int):
    return db.query(FamilyMember).filter(FamilyMember.familyMemberId == member_id).first()

def get_family_member_with_persons(db: Session, member_id: int):
    # query to get family member with related persons
    member = db.query(FamilyMember).filter(FamilyMember.familyMemberId == member_id).first()
    if member:
        return member, member.persons  # using backref relationship
    return None, []

def update_family_member(db: Session, member_id: int, **kwargs):
    member = db.query(FamilyMember).filter(FamilyMember.familyMemberId == member_id).first()
    if member:
        for key, value in kwargs.items():
            setattr(member, key, value)
        db.commit()
        db.refresh(member)
    return member

def delete_family_member(db: Session, member_id: int):
    # this function deletes a family member and all related persons
    try:
        # search for the family member
        member = db.query(FamilyMember).filter(FamilyMember.familyMemberId == member_id).first()
        if not member:
            return None
        
        # search for all related persons
        related_persons = db.query(Person).filter(Person.relatTo_id == member_id).all()
        
        # delete all related persons
        for person in related_persons:
            db.delete(person)
        
        # delete the family member
        db.delete(member)
        
        # save changes
        db.commit()
        
        print(f"✅ Deleted family member '{member.firstName} {member.lastName}' and {len(related_persons)} related persons")
        return member
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error deleting family member and related persons: {e}")
        raise e

def get_family_members_count(db: Session):
    return db.query(FamilyMember).count()

def get_recent_family_members(db: Session, limit: int = 10):
    return db.query(FamilyMember).order_by(FamilyMember.familyMemberId.desc()).limit(limit).all()

# ======================================================
# 🧩 CameraDetectedPerson CRUD + Actions
# ======================================================
def create_camera_detected_person(db: Session):
    cdp = CameraDetectedPerson()
    db.add(cdp)
    db.commit()
    db.refresh(cdp)
    return cdp

def get_camera_detected_persons(db: Session, skip: int = 0, limit: int = 100):
    return db.query(CameraDetectedPerson).offset(skip).limit(limit).all()

def get_camera_detected_person_by_id(db: Session, cdp_id: int):
    return db.query(CameraDetectedPerson).filter(CameraDetectedPerson.cameraDetectedPersonId == cdp_id).first()

def get_cdp_with_relations(db: Session, cdp_id: int):
    # query to get CDP with related persons, last seen records, and search results
    cdp = db.query(CameraDetectedPerson).filter(CameraDetectedPerson.cameraDetectedPersonId == cdp_id).first()
    if cdp:
        return cdp, cdp.persons, cdp.last_seen_records, cdp.search_results  # ✅ باستخدام العلاقات العكسية
    return None, [], [], []

def update_camera_detected_person(db: Session, cdp_id: int, **kwargs):
    cdp = db.query(CameraDetectedPerson).filter(CameraDetectedPerson.cameraDetectedPersonId == cdp_id).first()
    if cdp:
        for key, value in kwargs.items():
            setattr(cdp, key, value)
        db.commit()
        db.refresh(cdp)
    return cdp

def delete_camera_detected_person(db: Session, cdp_id: int):
    cdp = db.query(CameraDetectedPerson).filter(CameraDetectedPerson.cameraDetectedPersonId == cdp_id).first()
    if cdp:
        db.delete(cdp)
        db.commit()
    return cdp

def get_potentially_lost_persons(db: Session):
    return db.query(CameraDetectedPerson).filter(CameraDetectedPerson.potentiallyLost == True).all()

def get_elderly_persons(db: Session):
    return db.query(CameraDetectedPerson).filter(CameraDetectedPerson.isElderly == True).all()

def get_disabled_persons(db: Session):
    return db.query(CameraDetectedPerson).filter(CameraDetectedPerson.isDisabled == True).all()

def add_last_seen(db: Session, cdp_id: int, location: str, coordinates: list, time: datetime = None):
    cdp = db.query(CameraDetectedPerson).filter(CameraDetectedPerson.cameraDetectedPersonId == cdp_id).first()
    if cdp:
        last_seen = LastSeen(CDPid=cdp_id, location=location, coordinates=coordinates, time=time or datetime.now())
        db.add(last_seen)
        db.commit()
        db.refresh(last_seen)
        return last_seen
    return None

def calculate_ai_attributes(db: Session, cdp_id: int):
    cdp = db.query(CameraDetectedPerson).filter(CameraDetectedPerson.cameraDetectedPersonId == cdp_id).first()
    if cdp:
        cdp.calculate_ai_attributes()
        db.commit()
        db.refresh(cdp)
        return cdp
    return None

# ======================================================
# 🧩 Person CRUD + Actions - UPDATED
# ======================================================
def create_person(db: Session, **kwargs):
    """إنشاء شخص - النسخة الجديدة مع العلاقات المنفصلة"""
    errors = validate_person_creation(db, kwargs)
    if errors:
        raise ValueError(", ".join(errors))
    
    # ✅ إنشاء الشخص مباشرة (بدون CDP إجباري)
    person = Person(**kwargs)
    db.add(person)
    db.commit()
    db.refresh(person)
    
    return person

def create_person_with_cdp(db: Session, **kwargs):
    """إنشاء شخص مع CDP مرتبط"""
    errors = validate_person_creation(db, kwargs)
    if errors:
        raise ValueError(", ".join(errors))
    
    # 1. إنشاء CDP أولاً
    cdp = CameraDetectedPerson()
    db.add(cdp)
    db.commit()
    db.refresh(cdp)
    
    # 2. إنشاء الشخص مع ربطه بالـ CDP
    person = Person(cameraDetectedPersonId=cdp.cameraDetectedPersonId, **kwargs)
    db.add(person)
    db.commit()
    db.refresh(person)
    
    return person

def create_person_with_ai(db: Session, **kwargs):
    """إنشاء شخص مع attributes الـ AI تلقائياً"""
    errors = validate_person_creation(db, kwargs)
    if errors:
        raise ValueError(", ".join(errors))
    
    # 1. إنشاء CameraDetectedPerson مع AI
    cdp = CameraDetectedPerson()
    cdp.calculate_ai_attributes()  # توليد الـ AI attributes
    db.add(cdp)
    db.commit()
    db.refresh(cdp)
    
    # 2. إنشاء الشخص مع ربطه بالـ CDP
    person = Person(cameraDetectedPersonId=cdp.cameraDetectedPersonId, **kwargs)
    db.add(person)
    db.commit()
    db.refresh(person)
    
    return person

def get_persons(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Person).offset(skip).limit(limit).all()

def get_person_by_id(db: Session, person_id: int):
    return db.query(Person).filter(Person.personId == person_id).first()

def get_person_with_relations(db: Session, person_id: int):
    """جلب شخص مع كل العلاقات المرتبطة"""
    person = db.query(Person).filter(Person.personId == person_id).first()
    if person:
        return person, person.camera_detected_person, person.relatTo, person.reportedBy
    return None, None, None, None

def update_person(db: Session, person_id: int, **kwargs):
    person = db.query(Person).filter(Person.personId == person_id).first()
    if person:
        for key, value in kwargs.items():
            setattr(person, key, value)
        db.commit()
        db.refresh(person)
    return person

def delete_person(db: Session, person_id: int): # remember we have indicat inheritance
    # this function deletes a person and related CameraDetectedPerson if exists
    try:
        person = db.query(Person).filter(Person.personId == person_id).first()
        if not person:
            return None
        
        # If person has related CDP, delete it too
        if person.cameraDetectedPersonId:
            cdp = db.query(CameraDetectedPerson).filter(
                CameraDetectedPerson.cameraDetectedPersonId == person.cameraDetectedPersonId
            ).first()
            if cdp:
                db.delete(cdp)
        
        db.delete(person)
        db.commit()
        
        print(f"✅ Deleted person '{person.firstName} {person.lastName}' and all related records")
        return person
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error deleting person: {e}")
        raise e

def get_lost_persons(db: Session):
    return db.query(Person).filter(Person.isLost == True).all()

def get_found_persons(db: Session):
    return db.query(Person).filter(Person.isLost == False).all()

def quick_search_persons(db: Session, first_name: str = None, last_name: str = None, 
                        age: int = None, id_number: str = None, is_male: bool = None):
    query = db.query(Person)
    
    if first_name:
        query = query.filter(Person.firstName.ilike(f"%{first_name}%"))
    if last_name:
        query = query.filter(Person.lastName.ilike(f"%{last_name}%"))
    if age:
        query = query.filter(Person.age == age)
    if id_number:
        query = query.filter(Person.idNumber.ilike(f"%{id_number}%"))
    if is_male is not None:
        query = query.filter(Person.isMale == is_male)
    
    return query.all()

def update_person_status(db: Session, person_id: int, is_lost: bool):
    person = db.query(Person).filter(Person.personId == person_id).first()
    if person:
        # update status
        person.isLost = is_lost
        db.commit()
        db.refresh(person)
        return {"message": "Status updated successfully"}
    return {"error": "Person not found"}

def calculate_ai_attributes_for_person(db: Session, person_id: int):
    """حساب attributes الـ AI لشخص معين"""
    person = db.query(Person).filter(Person.personId == person_id).first()
    
    if person and person.camera_detected_person:
        person.camera_detected_person.calculate_ai_attributes()
        db.commit()
        return person.camera_detected_person
    
    return None

def find_similar_persons_by_embedding(db: Session, target_person_id: int, similarity_threshold: float = 0.7):
    """البحث عن أشخاص متشابهين باستخدام الـ embedding"""
    target_person = db.query(Person).filter(Person.personId == target_person_id).first()
    
    if not target_person or not target_person.camera_detected_person:
        return []
    
    target_embedding = target_person.camera_detected_person.embedding
    all_persons = db.query(Person).all()
    
    similar_persons = []
    for person in all_persons:
        if (person.personId != target_person_id and 
            person.camera_detected_person and 
            person.camera_detected_person.embedding):
            
            # محاكاة حساب التشابه
            similarity = random.uniform(0.1, 0.95)
            if similarity >= similarity_threshold:
                similar_persons.append({
                    'person': person,
                    'similarity': similarity,
                    'reason': f"High embedding similarity: {similarity:.2f}"
                })
    
    return similar_persons

def search_persons_by_family(db: Session, family_member_id: int):
    return db.query(Person).filter(Person.relatTo_id == family_member_id).all()

def get_persons_count(db: Session):
    return db.query(Person).count()

def get_recent_persons(db: Session, limit: int = 10):
    return db.query(Person).order_by(Person.personId.desc()).limit(limit).all()

# ======================================================
# 🧩 LastSeen CRUD + Actions
# ======================================================
def create_last_seen(db: Session, cdp_id: int, location: str, coordinates: list, time: datetime = None):
    last_seen = LastSeen(CDPid=cdp_id, location=location, coordinates=coordinates, time=time or datetime.now())
    db.add(last_seen)
    db.commit()
    db.refresh(last_seen)
    return last_seen

def get_last_seen_records(db: Session, skip: int = 0, limit: int = 100):
    return db.query(LastSeen).offset(skip).limit(limit).all()

def get_last_seen_by_id(db: Session, last_seen_id: int):
    return db.query(LastSeen).filter(LastSeen.id == last_seen_id).first()

def update_last_seen(db: Session, last_seen_id: int, **kwargs):
    last_seen = db.query(LastSeen).filter(LastSeen.id == last_seen_id).first()
    if last_seen:
        for key, value in kwargs.items():
            setattr(last_seen, key, value)
        db.commit()
        db.refresh(last_seen)
    return last_seen

def delete_last_seen(db: Session, last_seen_id: int):
    last_seen = db.query(LastSeen).filter(LastSeen.id == last_seen_id).first()
    if last_seen:
        db.delete(last_seen)
        db.commit()
    return last_seen

def get_recent_sightings(db: Session, hours: int = 24):
    time_threshold = datetime.now() - timedelta(hours=hours)
    return db.query(LastSeen).filter(LastSeen.time >= time_threshold).all()

def get_sightings_by_location(db: Session, location: str):
    return db.query(LastSeen).filter(LastSeen.location.ilike(f"%{location}%")).all()

def get_last_seen_count(db: Session):
    return db.query(LastSeen).count()

# ======================================================
# 🧩 SearchData CRUD + Actions
# ======================================================
def create_search_data(db: Session, requested_by_id: int):
    search_data = SearchData(requestedBy_id=requested_by_id)
    db.add(search_data)
    db.commit()
    db.refresh(search_data)
    return search_data

def get_search_data(db: Session, skip: int = 0, limit: int = 100):
    return db.query(SearchData).offset(skip).limit(limit).all()

def get_search_data_by_id(db: Session, search_id: int):
    return db.query(SearchData).filter(SearchData.searchID == search_id).first()

def get_search_data_with_results(db: Session, search_id: int):
    """جلب بحث مع نتائجه"""
    search_data = db.query(SearchData).filter(SearchData.searchID == search_id).first()
    if search_data:
        return search_data, search_data.results  # ✅ باستخدام العلاقة العكسية
    return None, []

def update_search_data(db: Session, search_id: int, **kwargs):
    search_data = db.query(SearchData).filter(SearchData.searchID == search_id).first()
    if search_data:
        for key, value in kwargs.items():
            setattr(search_data, key, value)
        db.commit()
        db.refresh(search_data)
    return search_data

def delete_search_data(db: Session, search_id: int):
    search_data = db.query(SearchData).filter(SearchData.searchID == search_id).first()
    if search_data:
        db.delete(search_data)
        db.commit()
    return search_data

def update_search_status(db: Session, search_id: int, is_found: bool):
    search_data = db.query(SearchData).filter(SearchData.searchID == search_id).first()
    if search_data:
        search_data.isFound = is_found
        search_data.isProcessing = False
        db.commit()
        return search_data
    return None

def get_active_searches(db: Session):
    return db.query(SearchData).filter(SearchData.isProcessing == True).all()

def get_completed_searches(db: Session):
    return db.query(SearchData).filter(SearchData.isProcessing == False).all()

def get_successful_searches(db: Session):
    return db.query(SearchData).filter(SearchData.isFound == True).all()

def cancel_search(db: Session, search_id: int):
    search_data = db.query(SearchData).filter(SearchData.searchID == search_id).first()
    if search_data:
        if search_data.isProcessing:
            return {"error": "Cannot delete a processing search."}
        db.delete(search_data)
        db.commit()
        return {"message": "Search deleted successfully"}
    return {"error": "Search not found"}

def get_search_data_count(db: Session):
    return db.query(SearchData).count()

# ======================================================
# 🧩 ResultsList CRUD + Actions
# ======================================================
def create_results_list(db: Session, search_id: int, camera_detected_person_id: int, is_accepted: bool = False):
    results_list = ResultsList(searchID=search_id, cameraDetectedPersonId=camera_detected_person_id, isAccepted=is_accepted)
    db.add(results_list)
    db.commit()
    db.refresh(results_list)
    return results_list

def get_results_list(db: Session, skip: int = 0, limit: int = 100):
    return db.query(ResultsList).offset(skip).limit(limit).all()

def get_results_list_by_id(db: Session, result_id: int):
    return db.query(ResultsList).filter(ResultsList.id == result_id).first()

def update_results_list(db: Session, result_id: int, **kwargs):
    results_list = db.query(ResultsList).filter(ResultsList.id == result_id).first()
    if results_list:
        for key, value in kwargs.items():
            setattr(results_list, key, value)
        db.commit()
        db.refresh(results_list)
    return results_list

def delete_results_list(db: Session, result_id: int):
    results_list = db.query(ResultsList).filter(ResultsList.id == result_id).first()
    if results_list:
        db.delete(results_list)
        db.commit()
    return results_list

def get_accepted_results(db: Session):
    return db.query(ResultsList).filter(ResultsList.isAccepted == True).all()

def get_rejected_results(db: Session):
    return db.query(ResultsList).filter(ResultsList.isAccepted == False).all()

def get_results_by_search(db: Session, search_id: int):
    return db.query(ResultsList).filter(ResultsList.searchID == search_id).all()

def get_results_by_person(db: Session, person_id: int):
    return db.query(ResultsList).filter(ResultsList.cameraDetectedPersonId == person_id).all()

def get_results_list_count(db: Session):
    return db.query(ResultsList).count()


def search_ai(db: Session, target_embedding: list, similarity_threshold: float):
    # this is a mock function to simulate AI search based on embedding similarity
    import random
    all_persons = get_persons(db)
    # fetch only persons with embeddings
    persons_with_embedding = [
        p for p in all_persons 
        if p.camera_detected_person and p.camera_detected_person.embedding
    ]
    #  temporarily return up to 3 random persons with embeddings
    return random.sample(persons_with_embedding, min(3, len(persons_with_embedding)))