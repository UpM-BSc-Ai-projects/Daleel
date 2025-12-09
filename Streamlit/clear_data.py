# clear_data.py
from database import get_db
from models import *
from sqlalchemy import text

def clear_all_data():
    try:
        db = next(get_db())
        
        print("🔄 Clearing all data...")
        
        # List of tables in correct order (according to relationships)
        tables = [
            "results_list",
            "search_data", 
            "last_seen",
            "persons",
            "camera_detected_persons",
            "family_members",
            "security_staff", 
            "admins"
        ]
        
        # Temporarily disable foreign keys
        db.execute(text("PRAGMA foreign_keys = OFF"))
        
        # Delete data from all tables
        for table in tables:
            db.execute(text(f"DELETE FROM {table}"))
            print(f"✅ Deleted data from: {table}")
        
        # Re-enable foreign keys
        db.execute(text("PRAGMA foreign_keys = ON"))
        
        db.commit()
        print("✅ All data deleted successfully")
        
        # Add new basic data
        print("🔄 Adding basic data...")
        
        # Admin examples
        admin1 = Admin(
            name="Main Administrator",
            email="admin@system.com", 
            passwordHash="admin123",
            phoneNumber="0550000000"
        )
        admin2 = Admin(
            name="Support Admin",
            email="support@system.com", 
            passwordHash="support123",
            phoneNumber="0550000001"
        )
        db.add(admin1)
        db.add(admin2)
        db.commit()
        db.refresh(admin1)
        db.refresh(admin2)
        print(f"✅ Added admin: {admin1.name}")
        print(f"✅ Added admin: {admin2.name}")
        
        # Security Staff examples
        staff1 = SecurityStaff(
            name="First Officer",
            email="staff1@system.com",
            passwordHash="staff123", 
            phoneNumber="0551111111",
            admin_id=admin1.adminId
        )
        staff2 = SecurityStaff(
            name="Second Officer",
            email="staff2@system.com",
            passwordHash="staff456", 
            phoneNumber="0551111112",
            admin_id=admin1.adminId
        )
        db.add(staff1)
        db.add(staff2)
        db.commit()
        db.refresh(staff1)
        db.refresh(staff2)
        print(f"✅ Added security staff: {staff1.name}")
        print(f"✅ Added security staff: {staff2.name}")
        
        # Family Member examples
        family1 = FamilyMember(
            firstName="Ahmed",
            lastName="Mohammed",
            idNumber="1234567890",
            phoneNumber="0552222222",
            relation="Father"
        )
        family2 = FamilyMember(
            firstName="Fatima",
            lastName="Ali",
            idNumber="0987654321",
            phoneNumber="0552222223",
            relation="Mother"
        )
        db.add(family1)
        db.add(family2)
        db.commit()
        print(f"✅ Added family member: {family1.firstName} {family1.lastName}")
        print(f"✅ Added family member: {family2.firstName} {family2.lastName}")
        
        # Camera Detected Person examples
        cdp1 = CameraDetectedPerson()
        cdp1.calculate_ai_attributes()
        
        cdp2 = CameraDetectedPerson()
        cdp2.calculate_ai_attributes()
        
        db.add(cdp1)
        db.add(cdp2)
        db.commit()
        db.refresh(cdp1)
        db.refresh(cdp2)
        print(f"✅ Added camera detected person: {cdp1.cameraDetectedPersonId}")
        print(f"✅ Added camera detected person: {cdp2.cameraDetectedPersonId}")
        
        # Person examples
        person1 = Person(
            firstName="John",
            lastName="Smith",
            age=25,
            isMale=True,
            idNumber="1122334455",
            phoneNumber="0553333333",
            LastLocation="Mall Area",
            description="Last seen near food court",
            relatTo_id=family1.familyMemberId,
            reportedBy_id=staff1.staffId,
            cameraDetectedPersonId=cdp1.cameraDetectedPersonId
        )
        
        person2 = Person(
            firstName="Sarah",
            lastName="Johnson", 
            age=30,
            isMale=False,
            idNumber="5566778899",
            phoneNumber="0553333334",
            LastLocation="Parking Lot B",
            description="Elderly woman with red bag",
            relatTo_id=family2.familyMemberId,
            reportedBy_id=staff2.staffId,
            cameraDetectedPersonId=cdp2.cameraDetectedPersonId
        )
        
        db.add(person1)
        db.add(person2)
        db.commit()
        print(f"✅ Added person: {person1.firstName} {person1.lastName}")
        print(f"✅ Added person: {person2.firstName} {person2.lastName}")
        
        # Last Seen examples
        last_seen1 = LastSeen(
            location="Camera 25 - North Entrance",
            coordinates=[24.7136, 46.6753],
            CDPid=cdp1.cameraDetectedPersonId
        )
        
        last_seen2 = LastSeen(
            location="Camera 12 - Parking Area", 
            coordinates=[24.7140, 46.6760],
            CDPid=cdp2.cameraDetectedPersonId
        )
        
        db.add(last_seen1)
        db.add(last_seen2)
        db.commit()
        print(f"✅ Added last seen record: {last_seen1.location}")
        print(f"✅ Added last seen record: {last_seen2.location}")
        
        # Search Data examples
        search1 = SearchData(
            requestedBy_id=staff1.staffId
        )
        
        search2 = SearchData(
            requestedBy_id=staff2.staffId
        )
        
        db.add(search1)
        db.add(search2)
        db.commit()
        db.refresh(search1)
        db.refresh(search2)
        print(f"✅ Added search data: {search1.searchID}")
        print(f"✅ Added search data: {search2.searchID}")
        
        # Results List examples
        result1 = ResultsList(
            searchID=search1.searchID,
            cameraDetectedPersonId=cdp1.cameraDetectedPersonId,
            isAccepted=True
        )
        
        result2 = ResultsList(
            searchID=search2.searchID,
            cameraDetectedPersonId=cdp2.cameraDetectedPersonId, 
            isAccepted=False
        )
        
        db.add(result1)
        db.add(result2)
        db.commit()
        print(f"✅ Added result: {result1.id}")
        print(f"✅ Added result: {result2.id}")
        
        print("🎉 Process completed! Data is ready for use")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    clear_all_data()