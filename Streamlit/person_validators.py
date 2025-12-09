from sqlalchemy.orm import Session
from models import Person

def validate_person_creation(db: Session, data: dict):
    errors = []
    
    # Validate phone number
    if data.get('phoneNumber'):
        existing = db.query(Person).filter(Person.phoneNumber == data['phoneNumber']).first()
        if existing:
            if existing.relatTo:
                relative_name = f"{existing.relatTo.firstName} {existing.relatTo.lastName}"
                relative_phone = existing.relatTo.phoneNumber
                relative_relation = existing.relatTo.relation if hasattr(existing.relatTo, 'relation') else 'relative'
                errors.append(f"This phone number is already recorded by his/her {relative_relation} . Whose name and phone {relative_name} ({relative_phone}).")
            elif existing.reportedBy:
                officer_name = f"{existing.reportedBy.name}"
                last_location = getattr(existing, 'LastLocation', 'unknown location')
                officer_phone = existing.reportedBy.phoneNumber
                errors.append(f"This phone number is already recorded by officer {officer_name} ({officer_phone}). He is in {last_location}.")
            else:
                errors.append("This phone number is already used.")
    
    # Validate ID number
    if data.get('idNumber'):
        existing = db.query(Person).filter(Person.idNumber == data['idNumber']).first()
        if existing:
            if existing.relatTo:
                relative_name = f"{existing.relatTo.firstName} {existing.relatTo.lastName}"
                relative_phone = existing.relatTo.phoneNumber
                relative_relation = existing.relatTo.relation if hasattr(existing.relatTo, 'relation') else 'relative'
                errors.append(f"This ID number is already recorded by his/her {relative_relation} . Whose name and phone {relative_name} ({relative_phone}).")
            elif existing.reportedBy:
                officer_name = f"{existing.reportedBy.name}"
                last_location = getattr(existing, 'LastLocation', 'unknown location')
                officer_phone = existing.reportedBy.phoneNumber
                errors.append(f"This ID number is already recorded by officer {officer_name} ({officer_phone}). He is in {last_location}.")
            else:
                errors.append("This ID number is already used.")
    
    # Validate name, gender
    first_name = data.get('firstName')
    last_name = data.get('lastName')
    is_male = data.get('isMale')
    
    if first_name and last_name and is_male is not None:
        
        # Validate exact match
        existing_person = db.query(Person).filter(
            Person.firstName == first_name,
            Person.lastName == last_name,
            Person.isMale == is_male
        ).first()
        
        if existing_person:
            if existing_person.relatTo:
                relative_name = f"{existing_person.relatTo.firstName} {existing_person.relatTo.lastName}"
                relative_phone = existing_person.relatTo.phoneNumber
                relative_relation = existing_person.relatTo.relation if hasattr(existing_person.relatTo, 'relation') else 'relative'
                errors.append(f"A person with the same name and gender already exists, reported by his/her {relative_relation}. Whose name and phone {relative_name} ({relative_phone}).")
            elif existing_person.reportedBy:
                officer_name = f"{existing_person.reportedBy.name}"
                last_location = getattr(existing_person, 'LastLocation', 'unknown location')
                officer_phone = existing_person.reportedBy.phoneNumber
                errors.append(f"A person with the same name and gender already exists, reported by officer {officer_name} ({officer_phone}). He is in {last_location}.")
            else:
                errors.append("A person with the same name and gender already exists.")
    
    return errors