import streamlit as st
import pandas as pd
import time
import os
from PIL import Image
from database import get_db
from crud import *

st.title("Person Management")

# Get database session
db = next(get_db())

# Create tabs for different functions
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Persons List", 
    "➕ Add Person", 
    "🔍 Quick Search", 
    "✏️ Update Person", 
    "🎯 Special Actions",
    "📊 Statistics"
])

with tab1:
    st.header("Persons List")
    persons = get_persons(db)

    if persons:
        data = []
        for person in persons:
            data.append({
                'ID': person.personId,
                'Name': f'{person.firstName} {person.lastName}',
                'Age': person.age,
                'Gender': 'Male' if person.isMale else 'Female',
                'ID Number': person.idNumber,
                'Phone': person.phoneNumber,
                'Location': person.LastLocation,
                'Status': 'Lost' if person.isLost else 'Found'
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Show person details with relations
        st.subheader("Person Details with Media")
        person_ids = [person.personId for person in persons]
        selected_person_id = st.selectbox("Select Person to view details:", person_ids)
        
        if selected_person_id:
            person, cdp, family, reporter = get_person_with_relations(db, selected_person_id)
            if person:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**First Name:** {person.firstName}")
                    st.write(f"**Last Name:** {person.lastName}")
                    st.write(f"**Age:** {person.age}")
                    st.write(f"**Gender:** {'Male' if person.isMale else 'Female'}")
                    st.write(f"**ID Number:** {person.idNumber}")
                    st.write(f"**Phone:** {person.phoneNumber}")
                    st.write(f"**Location:** {person.LastLocation}")
                
                with col2:
                    st.write(f"**Status:** {'Lost' if person.isLost else 'Found'}")
                    st.write(f"**Upload Time:** {person.uploadTime}")
                    
                    if family:
                        st.write(f"**Reported by Family:** {family.firstName} {family.lastName} ({family.relation})")
                    if reporter:
                        st.write(f"**Reported by Staff:** {reporter.name}")
                    if cdp:
                        st.write(f"**AI Attributes:** cdp_id: {cdp.cameraDetectedPersonId}, Potentially Lost: {cdp.potentiallyLost}, Elderly: {cdp.isElderly}, Disabled: {cdp.isDisabled}")
                
                if person.description:
                    st.write(f"**Description:** {person.description}")
                
                # images display
                st.subheader("📸 Person Images")
                images = []
                image_fields = ['image_1', 'image_2', 'image_3', 'image_4', 'image_5']
                
                for i, field in enumerate(image_fields, 1):
                    image_path = getattr(person, field, None)
                    if image_path and os.path.exists(image_path):
                        images.append((f"Image {i}", image_path))
                
                if images:
                    # show images in columns
                    cols = st.columns(len(images))
                    for idx, (col, (img_name, img_path)) in enumerate(zip(cols, images)):
                        with col:
                            st.write(f"**{img_name}**")
                            try:
                                image = Image.open(img_path)
                                st.image(image, use_container_width=True, caption=img_name)
                            except Exception as e:
                                st.error(f"Error loading image: {e}")
                else:
                    st.info("No images available for this person")
                
                # show video
                st.subheader("🎥 Person Video")
                if person.video and os.path.exists(person.video):
                    # show video player
                    st.video(person.video)
                    
                    # show video file size and path
                    video_stats = os.stat(person.video)
                    st.write(f"**Video Size:** {video_stats.st_size / (1024*1024):.2f} MB")
                    st.write(f"**Video Path:** {person.video}")
                else:
                    st.info("No video available for this person")
    else:
        st.info("No person data available.")

with tab2:
    st.header("Add New Person")
    
    # Get available family members and staff for selection
    family_members = get_family_members(db)
    staff_members = get_security_staffs(db)
    
    with st.form("add_person_form"):
        st.info("🤖 AI analysis will be automatically applied to all new persons")
        
        # Personal Information
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input("First Name*", max_chars=100, 
                                      help="Maximum 100 characters")
            last_name = st.text_input("Last Name*", max_chars=100,
                                    help="Maximum 100 characters")
            age = st.number_input("Age", min_value=0, max_value=120, value=0,
                                help="0 means unknown")
            is_male = st.selectbox("Gender*", [True, False], 
                                 format_func=lambda x: "Male" if x else "Female")
        
        with col2:
            id_number = st.text_input("ID Number", max_chars=25,
                                     help="Maximum 25 characters")
            phone_number = st.text_input("Phone Number", max_chars=20,
                                        help="Maximum 20 characters")
            last_location = st.text_input("Last Known Location", value="Unknown", 
                                        max_chars=255,
                                        help="Maximum 255 characters")
            description = st.text_area("Description", 
                                     help="Additional details about the person")
        
        # Relationships
        st.subheader("Relationships")
        col3, col4 = st.columns(2)
        with col3:
            family_options = {None: "No Family Member"}
            if family_members:
                family_options.update({fm.familyMemberId: f"{fm.firstName} {fm.lastName} ({fm.relation})" for fm in family_members})
            
            relatTo_id = st.selectbox("Reported by Family Member", 
                                    options=list(family_options.keys()),
                                    format_func=lambda x: family_options[x])
        
        with col4:
            staff_options = {None: "No Security Staff"}
            if staff_members:
                staff_options.update({s.staffId: f"{s.name}" for s in staff_members})
            
            reportedBy_id = st.selectbox("Reported by Security Staff", 
                                       options=list(staff_options.keys()),
                                       format_func=lambda x: staff_options[x])
        
        # Media Upload
        st.subheader("Media Upload")
        
        # Images - Maximum 5
        st.write("**Images** (Maximum 5 images)")
        uploaded_images = st.file_uploader(
            "Upload person images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="person_images"
        )
        
        if uploaded_images and len(uploaded_images) > 5:
            st.error("❌ Maximum 5 images allowed! Using first 5 images.")
            uploaded_images = uploaded_images[:5]
        
        # Video - Only One
        st.write("**Video** (One video only)")
        uploaded_video = st.file_uploader(
            "Upload person video",
            type=["mp4", "avi", "mov"],
            accept_multiple_files=False,
            key="person_video"
        )
        
        submitted = st.form_submit_button("Add Person")
        if submitted:
            # Validation
            if not first_name or not last_name:
                st.error("Please fill in all required fields (First Name, Last Name)")
            elif len(first_name) > 100:
                st.error("First name must be 100 characters or less")
            elif len(last_name) > 100:
                st.error("Last name must be 100 characters or less")
            elif id_number and len(id_number) > 25:
                st.error("ID number must be 25 characters or less")
            elif phone_number and len(phone_number) > 20:
                st.error("Phone number must be 20 characters or less")
            elif len(last_location) > 255:
                st.error("Location must be 255 characters or less")
            else:
                try:
                    # Create uploads directory if not exists
                    if not os.path.exists("uploads"):
                        os.makedirs("uploads")
                    
                    # Process and save images
                    image_paths = [None, None, None, None, None]
                    if uploaded_images:
                        for i, uploaded_file in enumerate(uploaded_images):
                            if i >= 5:  # Ensure max 5 images
                                break
                            file_path = f"uploads/person_{int(time.time())}_{i}_{uploaded_file.name}"
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            image_paths[i] = file_path
                    
                    # Process and save video
                    video_path = None
                    if uploaded_video:
                        video_path = f"uploads/person_{int(time.time())}_video_{uploaded_video.name}"
                        with open(video_path, "wb") as f:
                            f.write(uploaded_video.getbuffer())
                    
                    # Prepare person data
                    person_data = {
                        'firstName': first_name,
                        'lastName': last_name,
                        'age': age if age > 0 else None,
                        'isMale': is_male,
                        'idNumber': id_number or None,
                        'phoneNumber': phone_number or None,
                        'LastLocation': last_location,
                        'description': description or None,
                        'relatTo_id': relatTo_id if relatTo_id != None else None,
                        'reportedBy_id': reportedBy_id if reportedBy_id != None else None,
                        'video': video_path,
                        'image_1': image_paths[0],
                        'image_2': image_paths[1],
                        'image_3': image_paths[2],
                        'image_4': image_paths[3],
                        'image_5': image_paths[4]
                    }
                    
                    # ✅ Always use AI creation method
                    person = create_person_with_ai(db, **person_data)
                    
                    success_placeholder = st.empty()
                    success_placeholder.success("✅ Person added successfully with AI analysis!")
                    time.sleep(3)
                    success_placeholder.empty()
                    st.rerun()
                    
                except ValueError as e:
                    st.error(f"Validation Error: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab3:
    st.header("Quick Search")
    with st.form("quick_search"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_first = st.text_input("First Name")
            search_age = st.number_input("Age", min_value=0, max_value=120, value=0)
        
        with col2:
            search_last = st.text_input("Last Name")
            search_id = st.text_input("ID Number")
        
        with col3:
            search_gender = st.selectbox("Gender", [None, True, False], 
                                       format_func=lambda x: "All" if x is None else "Male" if x else "Female")
            search_status = st.selectbox("Status", [None, True, False],
                                       format_func=lambda x: "All" if x is None else "Lost" if x else "Found")
        
        search_clicked = st.form_submit_button("Search")
        
        if search_clicked:
            # البحث الأساسي بدون فلتر العمر أولاً
            results = quick_search_persons(db, search_first or None, search_last or None, 
                                         None,  # لا نمرر العمر هنا
                                         search_id or None, search_gender)
            
            # تطبيق فلتر العمر يدوياً (+2 / -2)
            if search_age > 0:
                filtered_results = []
                for person in results:
                    if person.age and (search_age - 2 <= person.age <= search_age + 2):
                        filtered_results.append(person)
                results = filtered_results
            
            # Filter by status if selected
            if search_status is not None:
                results = [p for p in results if p.isLost == search_status]
            
            if results:
                success_placeholder = st.empty()
                success_placeholder.success(f"✅ Found {len(results)} results")
                time.sleep(3)
                success_placeholder.empty()
                
                df_results = pd.DataFrame([{
                    'ID': p.personId,
                    'Name': f"{p.firstName} {p.lastName}",
                    'Age': p.age,
                    'Gender': 'Male' if p.isMale else 'Female',
                    'ID Number': p.idNumber,
                    'Phone': p.phoneNumber,
                    'Location': p.LastLocation,
                    'Status': 'Lost' if p.isLost else 'Found'
                } for p in results])
                st.dataframe(df_results, use_container_width=True)
            else:
                st.warning("No results found")

with tab4:
    st.header("Update Person Information")
    
    persons = get_persons(db)
    if persons:
        person_options = {f"{p.firstName} {p.lastName} (ID: {p.personId})": p.personId for p in persons}
        selected_person = st.selectbox("Select Person to Update:", list(person_options.keys()))
        
        if selected_person:
            person_id = person_options[selected_person]
            person = get_person_by_id(db, person_id)
            
            if person:
                with st.form("update_person_form"):
                    st.write("### Current Information")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.text_input("Person ID*", value=person.personId, disabled=True)
                        first_name = st.text_input("First Name*", value=person.firstName, max_chars=100)
                        last_name = st.text_input("Last Name*", value=person.lastName, max_chars=100)
                        age = st.number_input("Age", value=person.age or 0, min_value=0, max_value=120)
                        is_male = st.selectbox("Gender*", [True, False], 
                                             index=0 if person.isMale else 1,
                                             format_func=lambda x: "Male" if x else "Female")
                    
                    with col2:
                        id_number = st.text_input("ID Number", value=person.idNumber or "", max_chars=25)
                        phone_number = st.text_input("Phone Number", value=person.phoneNumber or "", max_chars=20)
                        last_location = st.text_input("Last Known Location", value=person.LastLocation, max_chars=255)
                        is_lost = st.selectbox("Status*", [True, False],
                                             index=0 if person.isLost else 1,
                                             format_func=lambda x: "Lost" if x else "Found")
                    
                    description = st.text_area("Description", value=person.description or "")
                    
                    # العلاقات - يمكن تعديلها
                    st.subheader("Relationships")
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        family_members = get_family_members(db)
                        family_options = {None: "No Family Member"}
                        if family_members:
                            family_options.update({fm.familyMemberId: f"{fm.firstName} {fm.lastName} ({fm.relation})" for fm in family_members})
                        
                        current_family = person.relatTo_id if person.relatTo_id else None
                        relatTo_id = st.selectbox("Family Member", 
                                                options=list(family_options.keys()),
                                                index=list(family_options.keys()).index(current_family) if current_family in family_options else 0,
                                                format_func=lambda x: family_options[x])
                    
                    with col4:
                        staff_members = get_security_staffs(db)
                        staff_options = {None: "No Security Staff"}
                        if staff_members:
                            staff_options.update({s.staffId: f"{s.name}" for s in staff_members})
                        
                        current_staff = person.reportedBy_id if person.reportedBy_id else None
                        reportedBy_id = st.selectbox("Security Staff", 
                                                   options=list(staff_options.keys()),
                                                   index=list(staff_options.keys()).index(current_staff) if current_staff in staff_options else 0,
                                                   format_func=lambda x: staff_options[x])
                    
                    # تحديث الوسائط
                    st.subheader("Update Media")
                    
                    # الصور الحالية
                    st.write("**Current Images:**")
                    current_images = []
                    image_fields = ['image_1', 'image_2', 'image_3', 'image_4', 'image_5']
                    for i, field in enumerate(image_fields, 1):
                        image_path = getattr(person, field, None)
                        if image_path:
                            current_images.append((f"Image {i}", image_path))
                    
                    if current_images:
                        cols = st.columns(len(current_images))
                        for idx, (col, (img_name, img_path)) in enumerate(zip(cols, current_images)):
                            with col:
                                st.write(f"**{img_name}**")
                                if os.path.exists(img_path):
                                    try:
                                        image = Image.open(img_path)
                                        st.image(image, use_container_width=True, caption=img_name)
                                    except:
                                        st.error("Error loading image")
                                else:
                                    st.warning("Image not found")
                    
                    # رفع صور جديدة
                    st.write("**Upload New Images** (Maximum 5 images)")
                    uploaded_images = st.file_uploader(
                        "Select new images",
                        type=["png", "jpg", "jpeg"],
                        accept_multiple_files=True,
                        key="update_person_images"
                    )
                    
                    if uploaded_images and len(uploaded_images) > 5:
                        st.error("❌ Maximum 5 images allowed! Using first 5 images.")
                        uploaded_images = uploaded_images[:5]
                    
                    # الفيديو الحالي
                    st.write("**Current Video:**")
                    if person.video and os.path.exists(person.video):
                        st.video(person.video)
                        st.write(f"Current video: {os.path.basename(person.video)}")
                    else:
                        st.info("No video available")
                    
                    # رفع فيديو جديد
                    st.write("**Upload New Video** (One video only)")
                    uploaded_video = st.file_uploader(
                        "Select new video",
                        type=["mp4", "avi", "mov"],
                        accept_multiple_files=False,
                        key="update_person_video"
                    )
                    
                    submitted = st.form_submit_button("Update Person")
                    if submitted:
                        if not first_name or not last_name:
                            st.error("Please fill in all required fields (First Name, Last Name)")
                        else:
                            try:
                                # معالجة الصور الجديدة
                                image_paths = [None, None, None, None, None]
                                if uploaded_images:
                                    for i, uploaded_file in enumerate(uploaded_images):
                                        if i >= 5:
                                            break
                                        file_path = f"uploads/person_update_{int(time.time())}_{i}_{uploaded_file.name}"
                                        with open(file_path, "wb") as f:
                                            f.write(uploaded_file.getbuffer())
                                        image_paths[i] = file_path
                                else:
                                    # الاحتفاظ بالصور القديمة إذا لم يتم رفع جديدة
                                    image_paths = [
                                        person.image_1, person.image_2, person.image_3, 
                                        person.image_4, person.image_5
                                    ]
                                
                                # معالجة الفيديو الجديد
                                video_path = person.video  # الاحتفاظ بالفيديو القديم
                                if uploaded_video:
                                    video_path = f"uploads/person_update_{int(time.time())}_video_{uploaded_video.name}"
                                    with open(video_path, "wb") as f:
                                        f.write(uploaded_video.getbuffer())
                                
                                update_person(db, person_id,
                                            firstName=first_name,
                                            lastName=last_name,
                                            age=age if age > 0 else None,
                                            isMale=is_male,
                                            idNumber=id_number or None,
                                            phoneNumber=phone_number or None,
                                            LastLocation=last_location,
                                            description=description or None,
                                            isLost=is_lost,
                                            relatTo_id=relatTo_id if relatTo_id != None else None,
                                            reportedBy_id=reportedBy_id if reportedBy_id != None else None,
                                            video=video_path,
                                            image_1=image_paths[0],
                                            image_2=image_paths[1],
                                            image_3=image_paths[2],
                                            image_4=image_paths[3],
                                            image_5=image_paths[4])
                                
                                success_placeholder = st.empty()
                                success_placeholder.success("✅ Person updated successfully!")
                                time.sleep(3)
                                success_placeholder.empty()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
            else:
                st.error("Person not found!")
    else:
        st.info("No persons available to update.")

with tab5:
    st.header("Special Actions")
    
    persons = get_persons(db)
    if persons:
        person_ids = [p.personId for p in persons]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Update Status")
            selected_person = st.selectbox("Select Person", person_ids, 
                                         format_func=lambda x: f"Person {x}")
            new_status = st.selectbox("New Status", [True, False], 
                                    format_func=lambda x: "Lost" if x else "Found")
            if st.button("Update Status"):
                result = update_person_status(db, selected_person, new_status)
                if "error" in result:
                    st.error(result["error"])
                else:
                    success_placeholder = st.empty()
                    success_placeholder.success("✅ " + result["message"])
                    time.sleep(3)
                    success_placeholder.empty()
                    st.rerun()
            
            st.subheader("Delete Found Person")
            # عرض الأشخاص الموجودين فقط
            found_persons = [p for p in persons if not p.isLost]
            
            if found_persons:
                # إنشاء قائمة بالأشخاص الموجودين مع عرض معلومات العائلة
                found_options = {}
                for person in found_persons:
                    family_info = ""
                    if person.relatTo:
                        family_info = f" - Family: {person.relatTo.firstName} {person.relatTo.lastName} (ID: {person.relatTo.familyMemberId})"
                    found_options[person.personId] = f"Person {person.personId}{family_info}"
                
                delete_person_id = st.selectbox("Select Found Person to Delete", 
                                              options=list(found_options.keys()),
                                              format_func=lambda x: found_options[x],
                                              key="delete_person")
                
                if delete_person_id:
                    person_to_delete = get_person_by_id(db, delete_person_id)
                    
                    if person_to_delete and person_to_delete.relatTo:
                        st.warning(f"**⚠️ This will also delete the related family member:**")
                        st.write(f"**Family ID:** {person_to_delete.relatTo.familyMemberId}")
                        st.write(f"**Family Name:** {person_to_delete.relatTo.firstName} {person_to_delete.relatTo.lastName}")
                        st.write(f"**Relation:** {person_to_delete.relatTo.relation}")
                    
                    confirmation = st.text_input("Type 'DELETE' to confirm:")
                    if st.button("Delete Person and Family", type="primary"):
                        if confirmation == "DELETE":
                            try:
                                # حذف العائلة أولاً إذا موجودة
                                if person_to_delete and person_to_delete.relatTo_id:
                                    delete_family_member(db, person_to_delete.relatTo_id)
                                
                                delete_person(db, delete_person_id)
                                success_placeholder = st.empty()
                                success_placeholder.success("✅ Person and related family deleted successfully!")
                                time.sleep(3)
                                success_placeholder.empty()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                        else:
                            st.error("Please type 'DELETE' to confirm deletion")
            else:
                st.info("No found persons available to delete")
        
        with col2:
            st.subheader("Update Last Location")
            location_person_id = st.selectbox("Select Person", person_ids,
                                            format_func=lambda x: f"Person {x}", key="location_person")
            new_location = st.text_input("New Location", max_chars=255)
            
            if st.button("Update Location"):
                if new_location:
                    try:
                        update_person(db, location_person_id, LastLocation=new_location)
                        success_placeholder = st.empty()
                        success_placeholder.success("✅ Location updated successfully!")
                        time.sleep(3)
                        success_placeholder.empty()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Please enter a location")
            
            st.subheader("Regenerate AI Analysis")
            ai_person = st.selectbox("Select Person for AI Regeneration", person_ids, 
                                   format_func=lambda x: f"Person {x}", key="ai_select")
            
            if st.button("Regenerate AI Analysis", type="primary"):
                try:
                    # إنشاء CDP جديد
                    new_cdp = create_camera_detected_person(db)
                    new_cdp.calculate_ai_attributes()
                    
                    # تحديث الشخص بالـ CDP الجديد
                    update_person(db, ai_person, cameraDetectedPersonId=new_cdp.cameraDetectedPersonId)
                    
                    success_placeholder = st.empty()
                    success_placeholder.success("✅ AI analysis regenerated successfully!")
                    time.sleep(3)
                    success_placeholder.empty()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    else:
        st.info("No persons available for special actions")

with tab6:
    st.header("Persons Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_count = get_persons_count(db)
        st.metric("Total Persons", total_count)
    
    with col2:
        lost_count = len(get_lost_persons(db))
        st.metric("Lost Persons", lost_count)
    
    with col3:
        found_count = len(get_found_persons(db))
        st.metric("Found Persons", found_count)
    
    # Gender distribution
    st.subheader("Gender Distribution")
    persons = get_persons(db)
    if persons:
        male_count = sum(1 for p in persons if p.isMale)
        female_count = total_count - male_count
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Male", male_count)
        with col2:
            st.metric("Female", female_count)
    
    # Recent persons table
    st.subheader("Recent Persons")
    recent_persons = get_recent_persons(db, 10)
    if recent_persons:
        recent_data = []
        for person in recent_persons:
            recent_data.append({
                'ID': person.personId,
                'Name': f"{person.firstName} {person.lastName}",
                'Age': person.age,
                'Gender': 'Male' if person.isMale else 'Female',
                'Status': 'Lost' if person.isLost else 'Found',
                'Location': person.LastLocation,
                'Upload Time': person.uploadTime
            })
        
        df_recent = pd.DataFrame(recent_data)
        st.dataframe(df_recent, use_container_width=True)
    else:
        st.info("No recent person data available.")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:**\n- Use different creation methods based on available data\n- AI analysis requires camera detection data\n- Check for duplicates using the validation system\n- Update status to 'Found' to automatically remove records")
