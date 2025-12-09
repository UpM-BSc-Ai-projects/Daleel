import streamlit as st
import pandas as pd
import time
from database import get_db
from crud import *

st.title("Security Staff Management")

# Get database session
db = next(get_db())

# Create tabs for different functions
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Staff List", 
    "➕ Add Staff", 
    "✏️ Update Staff", 
    "🗑️ Delete Staff", 
    "📊 Statistics"
])

with tab1:
    st.header("Security Staff List")
    staff_list = get_security_staffs(db)

    if staff_list:
        data = []
        for staff in staff_list:
            data.append({
                'ID': staff.staffId,
                'Name': staff.name,
                'Email': staff.email,
                'Phone': staff.phoneNumber,
                'Admin ID': staff.admin_id
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Show staff details with relations
        st.subheader("Staff Details with Relations")
        staff_ids = [staff.staffId for staff in staff_list]
        selected_staff_id = st.selectbox("Select Staff to view details:", staff_ids)
        
        if selected_staff_id:
            staff, reported_persons, search_requests = get_security_staff_with_relations(db, selected_staff_id)
            if staff:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {staff.name}")
                    st.write(f"**Email:** {staff.email}")
                    st.write(f"**Phone:** {staff.phoneNumber}")
                    st.write(f"**Admin ID:** {staff.admin_id}")
                
                with col2:
                    st.write(f"**Reported Persons:** {len(reported_persons)}")
                    st.write(f"**Search Requests:** {len(search_requests)}")
                    
                    if reported_persons:
                        st.write("**Recent Reported Persons:**")
                        for person in reported_persons[:3]:  # Show first 3
                            st.write(f"- {person.firstName} {person.lastName}")
                    
                    if len(reported_persons) > 3:
                        st.info(f"... and {len(reported_persons) - 3} more persons")
    else:
        st.info("No security staff data available.")

with tab2:
    st.header("Add New Security Staff")
    with st.form("add_staff_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Name*", max_chars=100, 
                                help="Maximum 100 characters")
            email = st.text_input("Email*", max_chars=200,
                                 help="Maximum 200 characters")
        
        with col2:
            password_hash = st.text_input("Password Hash*", type="password", 
                                        max_chars=255,
                                        help="Maximum 255 characters")
            phone_number = st.text_input("Phone Number", max_chars=20,
                                        help="Maximum 20 characters")
        
        # Get available admins for selection
        admins = get_admins(db)
        admin_options = {0: "No Admin"}
        if admins:
            admin_options.update({admin.adminId: f"{admin.name} (ID: {admin.adminId})" for admin in admins})
        
        admin_id = st.selectbox("Assign to Admin*", 
                               options=list(admin_options.keys()),
                               format_func=lambda x: admin_options[x])
        
        submitted = st.form_submit_button("Add Security Staff")
        if submitted:
            # Validation
            if not name or not email or not password_hash:
                st.error("Please fill in all required fields (Name, Email, Password Hash)")
            elif len(name) > 100:
                st.error("Name must be 100 characters or less")
            elif len(email) > 200:
                st.error("Email must be 200 characters or less")
            elif len(password_hash) > 255:
                st.error("Password hash must be 255 characters or less")
            elif phone_number and len(phone_number) > 20:
                st.error("Phone number must be 20 characters or less")
            else:
                try:
                    # Convert admin_id to None if 0 (No Admin selected)
                    final_admin_id = admin_id if admin_id != 0 else None
                    create_security_staff(db, name, email, password_hash, phone_number, final_admin_id)
                    success_placeholder = st.empty()
                    success_placeholder.success("✅ Security staff added successfully!")
                    time.sleep(3)
                    success_placeholder.empty()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

with tab3:
    st.header("Update Security Staff Information")
    
    # Get all staff for selection
    staff_list = get_security_staffs(db)
    if staff_list:
        staff_options = {f"{staff.name} (ID: {staff.staffId})": staff.staffId for staff in staff_list}
        selected_staff = st.selectbox("Select Staff to Update:", list(staff_options.keys()))
        
        if selected_staff:
            staff_id = staff_options[selected_staff]
            staff = get_security_staff_by_id(db, staff_id)
            
            if staff:
                with st.form("update_staff_form"):
                    st.write("### Current Information")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Display non-editable fields
                        st.text_input("Staff ID*", value=staff.staffId, disabled=True)
                        name = st.text_input("Name*", value=staff.name, max_chars=100,
                                           help="Maximum 100 characters")
                        email = st.text_input("Email*", value=staff.email, max_chars=200,
                                            help="Maximum 200 characters")
                    
                    with col2:
                        password_hash = st.text_input("Password Hash*", value=staff.passwordHash,
                                                    max_chars=255, help="Maximum 255 characters")
                        phone_number = st.text_input("Phone Number", value=staff.phoneNumber or "",
                                                   max_chars=20, help="Maximum 20 characters")
                    
                    # Get available admins for selection
                    admins = get_admins(db)
                    admin_options = {None: "No Admin"}
                    if admins:
                        admin_options.update({admin.adminId: f"{admin.name} (ID: {admin.adminId})" for admin in admins})
                    
                    current_admin = staff.admin_id if staff.admin_id else None
                    admin_id = st.selectbox("Assign to Admin", 
                                           options=list(admin_options.keys()),
                                           index=list(admin_options.keys()).index(current_admin) if current_admin in admin_options else 0,
                                           format_func=lambda x: admin_options[x])
                    
                    submitted = st.form_submit_button("Update Security Staff")
                    if submitted:
                        # Validation
                        if not name or not email or not password_hash:
                            st.error("Please fill in all required fields (Name, Email, Password Hash)")
                        elif len(name) > 100:
                            st.error("Name must be 100 characters or less")
                        elif len(email) > 200:
                            st.error("Email must be 200 characters or less")
                        elif len(password_hash) > 255:
                            st.error("Password hash must be 255 characters or less")
                        elif phone_number and len(phone_number) > 20:
                            st.error("Phone number must be 20 characters or less")
                        else:
                            try:
                                update_security_staff(db, staff_id, 
                                                   name=name, 
                                                   email=email, 
                                                   passwordHash=password_hash, 
                                                   phoneNumber=phone_number,
                                                   admin_id=admin_id if admin_id != None else None)
                                success_placeholder = st.empty()
                                success_placeholder.success("✅ Security staff updated successfully!")
                                time.sleep(3)
                                success_placeholder.empty()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
            else:
                st.error("Security staff not found!")
    else:
        st.info("No security staff available to update.")

with tab4:
    st.header("Delete Security Staff")
    
    staff_list = get_security_staffs(db)
    if staff_list:
        staff_options = {f"{staff.name} (ID: {staff.staffId})": staff.staffId for staff in staff_list}
        selected_staff = st.selectbox("Select Staff to Delete:", list(staff_options.keys()))
        
        if selected_staff:
            staff_id = staff_options[selected_staff]
            staff = get_security_staff_by_id(db, staff_id)
            
            if staff:
                st.warning("⚠️ Danger Zone - This action cannot be undone!")
                
                # Show staff details before deletion
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {staff.name}")
                    st.write(f"**Email:** {staff.email}")
                    st.write(f"**Phone:** {staff.phoneNumber}")
                with col2:
                    st.write(f"**Admin ID:** {staff.admin_id}")
                
                # Confirmation مباشر بدون فحص العلاقات
                confirmation = st.text_input("Type 'DELETE' to confirm:")
                if st.button("Delete Security Staff", type="primary"):
                    if confirmation == "DELETE":
                        try:
                            delete_security_staff(db, staff_id)
                            success_placeholder = st.empty()
                            success_placeholder.success("✅ Security staff deleted successfully!")
                            time.sleep(3)
                            success_placeholder.empty()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.error("Please type 'DELETE' to confirm deletion")
    else:
        st.info("No security staff available to delete.")

with tab5:
    st.header("Security Staff Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_staff = get_security_staff_count(db)
        st.metric("Total Security Staff", total_staff)
    
    with col2:
        recent_staff = get_recent_security_staff(db, 5)
        st.metric("Recent Staff (Last 5)", len(recent_staff))
    
    with col3:
        all_staff = get_security_staffs(db)
        staff_with_reports = sum(1 for staff in all_staff if staff.reported_persons)
        st.metric("Staff with Reports", staff_with_reports)
    
    # Recent staff table
    st.subheader("Recent Security Staff")
    recent_staff = get_recent_security_staff(db, 10)
    if recent_staff:
        recent_data = []
        for staff in recent_staff:
            recent_data.append({
                'ID': staff.staffId,
                'Name': staff.name,
                'Email': staff.email,
                'Phone': staff.phoneNumber,
                'Admin ID': staff.admin_id,
                'Reports Count': len(staff.reported_persons)
            })
        
        df_recent = pd.DataFrame(recent_data)
        st.dataframe(df_recent, use_container_width=True)
    else:
        st.info("No recent security staff data available.")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:**\n- Use the tabs to navigate between different staff operations\n- PK/FK fields cannot be modified\n- Staff deletion is now direct without relation checks")