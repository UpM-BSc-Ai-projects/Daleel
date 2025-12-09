import streamlit as st
import pandas as pd
import time
from database import get_db
from crud import *

st.title("Admin Management")

# Get database session
db = next(get_db())

# Create tabs for different functions
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Admins List", 
    "➕ Add Admin", 
    "✏️ Update Admin", 
    "🗑️ Delete Admin", 
    "📊 Statistics"
])

with tab1:
    st.header("Admins List")
    admins = get_admins(db)

    if admins:
        data = []
        for admin in admins:
            data.append({
                'ID': admin.adminId,
                'Name': admin.name,
                'Email': admin.email,
                'Phone': admin.phoneNumber
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Show admin details with staff
        st.subheader("Admin Details with Staff")
        admin_ids = [admin.adminId for admin in admins]
        selected_admin_id = st.selectbox("Select Admin to view details:", admin_ids)
        
        if selected_admin_id:
            admin, staff_list = get_admin_with_staff(db, selected_admin_id)
            if admin:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {admin.name}")
                    st.write(f"**Email:** {admin.email}")
                    st.write(f"**Phone:** {admin.phoneNumber}")
                
                with col2:
                    st.write(f"**Total Staff:** {len(staff_list)}")
                    if staff_list:
                        staff_names = [staff.name for staff in staff_list]
                        st.write(f"**Staff Members:** {', '.join(staff_names)}")
                    else:
                        st.info("No staff members assigned to this admin")
    else:
        st.info("No admin data available.")

with tab2:
    st.header("Add New Admin")
    with st.form("add_admin_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Name*", max_chars=200, 
                                help="Maximum 200 characters")
            email = st.text_input("Email*", max_chars=200,
                                 help="Maximum 200 characters")
        
        with col2:
            password_hash = st.text_input("Password Hash*", type="password", 
                                        max_chars=255,
                                        help="Maximum 255 characters")
            phone_number = st.text_input("Phone Number", max_chars=20,
                                        help="Maximum 20 characters")
        
        submitted = st.form_submit_button("Add Admin")
        if submitted:
            # Validation
            if not name or not email or not password_hash:
                st.error("Please fill in all required fields (Name, Email, Password Hash)")
            elif len(name) > 200:
                st.error("Name must be 200 characters or less")
            elif len(email) > 200:
                st.error("Email must be 200 characters or less")
            elif len(password_hash) > 255:
                st.error("Password hash must be 255 characters or less")
            elif phone_number and len(phone_number) > 20:
                st.error("Phone number must be 20 characters or less")
            else:
                try:
                    create_admin(db, name, email, password_hash, phone_number)
                    success_placeholder = st.empty()
                    success_placeholder.success("✅ Admin added successfully!")
                    time.sleep(3)
                    success_placeholder.empty()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

with tab3:
    st.header("Update Admin Information")
    
    # Get all admins for selection
    admins = get_admins(db)
    if admins:
        admin_options = {f"{admin.name} (ID: {admin.adminId})": admin.adminId for admin in admins}
        selected_admin = st.selectbox("Select Admin to Update:", list(admin_options.keys()))
        
        if selected_admin:
            admin_id = admin_options[selected_admin]
            admin = get_admin_by_id(db, admin_id)
            
            if admin:
                with st.form("update_admin_form"):
                    st.write("### Current Information")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Display non-editable fields
                        st.text_input("Admin ID*", value=admin.adminId, disabled=True)
                        name = st.text_input("Name*", value=admin.name, max_chars=200,
                                           help="Maximum 200 characters")
                        email = st.text_input("Email*", value=admin.email, max_chars=200,
                                            help="Maximum 200 characters")
                    
                    with col2:
                        password_hash = st.text_input("Password Hash*", value=admin.passwordHash,
                                                    max_chars=255, help="Maximum 255 characters")
                        phone_number = st.text_input("Phone Number", value=admin.phoneNumber or "",
                                                   max_chars=20, help="Maximum 20 characters")
                    
                    submitted = st.form_submit_button("Update Admin")
                    if submitted:
                        # Validation
                        if not name or not email or not password_hash:
                            st.error("Please fill in all required fields (Name, Email, Password Hash)")
                        elif len(name) > 200:
                            st.error("Name must be 200 characters or less")
                        elif len(email) > 200:
                            st.error("Email must be 200 characters or less")
                        elif len(password_hash) > 255:
                            st.error("Password hash must be 255 characters or less")
                        elif phone_number and len(phone_number) > 20:
                            st.error("Phone number must be 20 characters or less")
                        else:
                            try:
                                update_admin(db, admin_id, 
                                           name=name, 
                                           email=email, 
                                           passwordHash=password_hash, 
                                           phoneNumber=phone_number)
                                success_placeholder = st.empty()
                                success_placeholder.success("✅ Admin updated successfully!")
                                time.sleep(3)
                                success_placeholder.empty()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
            else:
                st.error("Admin not found!")
    else:
        st.info("No admins available to update.")

with tab4:
    st.header("Delete Admin")
    
    admins = get_admins(db)
    if admins:
        admin_options = {f"{admin.name} (ID: {admin.adminId})": admin.adminId for admin in admins}
        selected_admin = st.selectbox("Select Admin to Delete:", list(admin_options.keys()))
        
        if selected_admin:
            admin_id = admin_options[selected_admin]
            admin = get_admin_by_id(db, admin_id)
            
            if admin:
                st.warning("⚠️ Danger Zone - This action cannot be undone!")
                
                # Show admin details before deletion
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {admin.name}")
                    st.write(f"**Email:** {admin.email}")
                with col2:
                    st.write(f"**Phone:** {admin.phoneNumber}")
                
                # Confirmation مباشر بدون فحص العلاقات
                confirmation = st.text_input("Type 'DELETE' to confirm:")
                if st.button("Delete Admin", type="primary"):
                    if confirmation == "DELETE":
                        try:
                            delete_admin(db, admin_id)
                            success_placeholder = st.empty()
                            success_placeholder.success("✅ Admin deleted successfully!")
                            time.sleep(3)
                            success_placeholder.empty()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.error("Please type 'DELETE' to confirm deletion")
    else:
        st.info("No admins available to delete.")

with tab5:
    st.header("Admin Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_admins = get_admins_count(db)
        st.metric("Total Admins", total_admins)
    
    with col2:
        recent_admins = get_recent_admins(db, 5)
        st.metric("Recent Admins (Last 5)", len(recent_admins))
    
    with col3:
        all_admins = get_admins(db)
        admins_with_staff = sum(1 for admin in all_admins if admin.security_staffs)
        st.metric("Admins with Staff", admins_with_staff)
    
    # Recent admins table
    st.subheader("Recent Admins")
    recent_admins = get_recent_admins(db, 10)
    if recent_admins:
        recent_data = []
        for admin in recent_admins:
            recent_data.append({
                'ID': admin.adminId,
                'Name': admin.name,
                'Email': admin.email,
                'Phone': admin.phoneNumber,
                'Staff Count': len(admin.security_staffs)
            })
        
        df_recent = pd.DataFrame(recent_data)
        st.dataframe(df_recent, use_container_width=True)
    else:
        st.info("No recent admin data available.")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:**\n- Use the tabs to navigate between different admin operations\n- PK/FK fields cannot be modified\n- Admin deletion is now direct without staff checks")