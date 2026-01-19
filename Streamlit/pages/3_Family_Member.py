import streamlit as st
import pandas as pd
import time
from database import get_db
from crud import *

st.title("Family Member Management")

# Get database session
db = next(get_db())

# Create tabs for different functions
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Family Members List", 
    "➕ Add Family Member", 
    "✏️ Update Family Member", 
    "🗑️ Delete Family Member", 
    "📊 Statistics"
])

with tab1:
    st.header("Family Members List")
    family_members = get_family_members(db)

    if family_members:
        data = []
        for member in family_members:
            data.append({
                'ID': member.familyMemberId,
                'First Name': member.firstName,
                'Last Name': member.lastName,
                'ID Number': member.idNumber,
                'Phone': member.phoneNumber,
                'Relation': member.relation
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Show family member details with persons
        st.subheader("Family Member Details with Related Persons")
        member_ids = [member.familyMemberId for member in family_members]
        selected_member_id = st.selectbox("Select Family Member to view details:", member_ids)
        
        if selected_member_id:
            member, persons_list = get_family_member_with_persons(db, selected_member_id)
            if member:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**First Name:** {member.firstName}")
                    st.write(f"**Last Name:** {member.lastName}")
                    st.write(f"**ID Number:** {member.idNumber}")
                
                with col2:
                    st.write(f"**Phone:** {member.phoneNumber}")
                    st.write(f"**Relation:** {member.relation}")
                    st.write(f"**Related Persons:** {len(persons_list)}")
                
                if persons_list:
                    st.write("**Related Missing Persons:**")
                    for person in persons_list:
                        status = "Lost" if person.isLost else "Found"
                        st.write(f"- {person.firstName} {person.lastName} (Age: {person.age}, Status: {status})")
                else:
                    st.info("No related persons found for this family member.")
    else:
        st.info("No family member data available.")

with tab2:
    st.header("Add New Family Member")
    with st.form("add_family_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input("First Name*", max_chars=100, 
                                      help="Maximum 100 characters")
            last_name = st.text_input("Last Name*", max_chars=100,
                                    help="Maximum 100 characters")
            id_number = st.text_input("ID Number*", max_chars=25,
                                     help="Maximum 25 characters")
        
        with col2:
            phone_number = st.text_input("Phone Number", max_chars=20,
                                        help="Maximum 20 characters")
            relation = st.selectbox("Relation*", 
                                   ["Father", "Mother", "Spouse", "Child", "Sibling", "Grandparent", "Other"],
                                   help="Relationship to the missing person")
        
        submitted = st.form_submit_button("Add Family Member")
        if submitted:
            # Validation
            if not first_name or not last_name or not id_number or not relation:
                st.error("Please fill in all required fields (First Name, Last Name, ID Number, Relation)")
            elif len(first_name) > 100:
                st.error("First name must be 100 characters or less")
            elif len(last_name) > 100:
                st.error("Last name must be 100 characters or less")
            elif len(id_number) > 25:
                st.error("ID number must be 25 characters or less")
            elif phone_number and len(phone_number) > 20:
                st.error("Phone number must be 20 characters or less")
            else:
                try:
                    create_family_member(db, first_name, last_name, id_number, phone_number, relation)
                    success_placeholder = st.empty()
                    success_placeholder.success("✅ Family member added successfully!")
                    time.sleep(3)
                    success_placeholder.empty()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

with tab3:
    st.header("Update Family Member Information")
    
    # Get all family members for selection
    family_members = get_family_members(db)
    if family_members:
        member_options = {f"{member.firstName} {member.lastName} (ID: {member.familyMemberId})": member.familyMemberId for member in family_members}
        selected_member = st.selectbox("Select Family Member to Update:", list(member_options.keys()))
        
        if selected_member:
            member_id = member_options[selected_member]
            member = get_family_member_by_id(db, member_id)
            
            if member:
                with st.form("update_family_form"):
                    st.write("### Current Information")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Display non-editable fields
                        st.text_input("Family Member ID*", value=member.familyMemberId, disabled=True)
                        first_name = st.text_input("First Name*", value=member.firstName, max_chars=100,
                                                 help="Maximum 100 characters")
                        last_name = st.text_input("Last Name*", value=member.lastName, max_chars=100,
                                                help="Maximum 100 characters")
                        id_number = st.text_input("ID Number*", value=member.idNumber, max_chars=25,
                                                help="Maximum 25 characters")
                    
                    with col2:
                        phone_number = st.text_input("Phone Number", value=member.phoneNumber or "",
                                                   max_chars=20, help="Maximum 20 characters")
                        relation = st.selectbox("Relation*", 
                                              ["Father", "Mother", "Spouse", "Child", "Sibling", "Grandparent", "Other"],
                                              index=["Father", "Mother", "Spouse", "Child", "Sibling", "Grandparent", "Other"].index(member.relation) if member.relation in ["Father", "Mother", "Spouse", "Child", "Sibling", "Grandparent", "Other"] else 6)
                    
                    submitted = st.form_submit_button("Update Family Member")
                    if submitted:
                        # Validation
                        if not first_name or not last_name or not id_number or not relation:
                            st.error("Please fill in all required fields (First Name, Last Name, ID Number, Relation)")
                        elif len(first_name) > 100:
                            st.error("First name must be 100 characters or less")
                        elif len(last_name) > 100:
                            st.error("Last name must be 100 characters or less")
                        elif len(id_number) > 25:
                            st.error("ID number must be 25 characters or less")
                        elif phone_number and len(phone_number) > 20:
                            st.error("Phone number must be 20 characters or less")
                        else:
                            try:
                                update_family_member(db, member_id, 
                                                   firstName=first_name, 
                                                   lastName=last_name, 
                                                   idNumber=id_number, 
                                                   phoneNumber=phone_number, 
                                                   relation=relation)
                                success_placeholder = st.empty()
                                success_placeholder.success("✅ Family member updated successfully!")
                                time.sleep(3)
                                success_placeholder.empty()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
            else:
                st.error("Family member not found!")
    else:
        st.info("No family members available to update.")

with tab4:
    st.header("Delete Family Member")
    
    family_members = get_family_members(db)
    if family_members:
        member_options = {f"{member.firstName} {member.lastName} (ID: {member.familyMemberId})": member.familyMemberId for member in family_members}
        selected_member = st.selectbox("Select Family Member to Delete:", list(member_options.keys()))
        
        if selected_member:
            member_id = member_options[selected_member]
            member = get_family_member_by_id(db, member_id)
            
            if member:
                st.warning("⚠️ Danger Zone - This action cannot be undone!")
                
                # Show family member details
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**First Name:** {member.firstName}")
                    st.write(f"**Last Name:** {member.lastName}")
                    st.write(f"**ID Number:** {member.idNumber}")
                with col2:
                    st.write(f"**Phone:** {member.phoneNumber}")
                    st.write(f"**Relation:** {member.relation}")
                
                # Show persons that will be deleted
                _, persons_list = get_family_member_with_persons(db, member_id)
                if persons_list:
                    st.error(f"🚨 {len(persons_list)} related persons will also be deleted:")
                    
                    persons_data = []
                    for person in persons_list:
                        persons_data.append({
                            'Person ID': person.personId,
                            'Name': f"{person.firstName} {person.lastName}",
                            'Age': person.age or 'Unknown',
                            'Status': 'Lost' if person.isLost else 'Found'
                        })
                    
                    df_persons = pd.DataFrame(persons_data)
                    st.dataframe(df_persons, use_container_width=True)
                
                # Delete confirmation
                confirmation = st.text_input("Type 'DELETE' to confirm:")
                if st.button("Delete Family Member", type="primary"):
                    if confirmation == "DELETE":
                        try:
                            delete_family_member(db, member_id)
                            success_placeholder = st.empty()
                            success_placeholder.success(f"✅ Family member and {len(persons_list)} related persons deleted successfully!")
                            time.sleep(3)
                            success_placeholder.empty()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.error("Please type 'DELETE' to confirm deletion")
    else:
        st.info("No family members available to delete.")

with tab5:
    st.header("Family Members Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_members = get_family_members_count(db)
        st.metric("Total Family Members", total_members)
    
    with col2:
        recent_members = get_recent_family_members(db, 5)
        st.metric("Recent Members (Last 5)", len(recent_members))
    
    with col3:
        all_members = get_family_members(db)
        members_with_persons = sum(1 for member in all_members if member.persons)
        st.metric("Members with Related Persons", members_with_persons)
    
    # Relation statistics
    st.subheader("Relation Distribution")
    if family_members:
        relation_counts = {}
        for member in family_members:
            relation = member.relation
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
        
        relation_data = []
        for relation, count in relation_counts.items():
            relation_data.append({
                'Relation': relation,
                'Count': count,
                'Percentage': f"{(count / total_members * 100):.1f}%"
            })
        
        df_relations = pd.DataFrame(relation_data)
        st.dataframe(df_relations, use_container_width=True)
    
    # Recent family members table
    st.subheader("Recent Family Members")
    recent_members = get_recent_family_members(db, 10)
    if recent_members:
        recent_data = []
        for member in recent_members:
            recent_data.append({
                'ID': member.familyMemberId,
                'First Name': member.firstName,
                'Last Name': member.lastName,
                'ID Number': member.idNumber,
                'Phone': member.phoneNumber,
                'Relation': member.relation,
                'Persons Count': len(member.persons)
            })
        
        df_recent = pd.DataFrame(recent_data)
        st.dataframe(df_recent, use_container_width=True)
    else:
        st.info("No recent family member data available.")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:**\n- Use the tabs to navigate between different operations\n- PK fields cannot be modified\n- Deleting a family member will also delete all related persons\n- Family members can report multiple missing persons")