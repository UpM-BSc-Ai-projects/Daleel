import streamlit as st
import pandas as pd
from database import get_db
from crud import *

st.title("Camera Detected Person Management")

# Get database session
db = next(get_db())

# Create tabs for different functions
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 CDP List", 
    "➕ Add Last Seen", 
    "🤖 AI Processing", 
    "🔍 Special Actions",
    "📊 Statistics"
])

with tab1:
    st.header("Camera Detected Persons List")
    cdp_list = get_camera_detected_persons(db)

    if cdp_list:
        data = []
        for cdp in cdp_list:
            data.append({
                'ID': cdp.cameraDetectedPersonId,
                # 'Embedding Length': len(cdp.embedding) if cdp.embedding else 0,
                'Potentially Lost': 'Yes' if cdp.potentiallyLost else 'No',
                'Is Elderly': 'Yes' if cdp.isElderly else 'No',
                'Is Disabled': 'Yes' if cdp.isDisabled else 'No'
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Show CDP details with relations
        st.subheader("CDP Details with Relations")
        cdp_ids = [cdp.cameraDetectedPersonId for cdp in cdp_list]
        selected_cdp_id = st.selectbox("Select CDP to view details:", cdp_ids)
        
        if selected_cdp_id:
            cdp, persons, last_seen_records, search_results = get_cdp_with_relations(db, selected_cdp_id)
            if cdp:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**CDP ID:** {cdp.cameraDetectedPersonId}")
                    # st.write(f"**Embedding Length:** {len(cdp.embedding) if cdp.embedding else 0}")
                    st.write(f"**Potentially Lost:** {'Yes' if cdp.potentiallyLost else 'No'}")
                    st.write(f"**Is Elderly:** {'Yes' if cdp.isElderly else 'No'}")
                    st.write(f"**Is Disabled:** {'Yes' if cdp.isDisabled else 'No'}")
                
                with col2:
                    st.write(f"**Related Persons:** {len(persons)}")
                    st.write(f"**Last Seen Records:** {len(last_seen_records)}")
                    st.write(f"**Search Results:** {len(search_results)}")
                
                # Show related persons
                if persons:
                    st.write("**Related Persons:**")
                    for person in persons:
                        st.write(f"- {person.firstName} {person.lastName} (ID: {person.personId})")
                
                # Show recent last seen records
                if last_seen_records:
                    st.write("**Recent Last Seen Records:**")
                    for record in last_seen_records[:3]:  # Show last 3
                        st.write(f"- {record.location} at {record.time}")
    else:
        st.info("No camera detected persons data available.")

with tab2:
    st.header("Add Last Seen Record")
    
    cdp_list = get_camera_detected_persons(db)
    if cdp_list:
        cdp_ids = [cdp.cameraDetectedPersonId for cdp in cdp_list]
        
        with st.form("add_last_seen_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                selected_cdp = st.selectbox("Select CDP*", cdp_ids,
                                          help="Select the camera detected person")
                location = st.text_input("Location*", value="Camera XXX", max_chars=255,
                                       help="Maximum 255 characters")
            
            with col2:
                coordinates_input = st.text_input("Coordinates (comma separated)*", 
                                                value="0.0,0.0",
                                                help="Enter coordinates as 'latitude,longitude'")
                time_note = st.info("Current time will be used automatically")
            
            submitted = st.form_submit_button("Add Last Seen")
            if submitted:
                # Validation
                if not location or not coordinates_input:
                    st.error("Please fill in all required fields (Location, Coordinates)")
                elif len(location) > 255:
                    st.error("Location must be 255 characters or less")
                else:
                    try:
                        # Parse coordinates
                        coords_list = [float(coord.strip()) for coord in coordinates_input.split(",")]
                        if len(coords_list) != 2:
                            st.error("Please enter exactly 2 coordinates (latitude, longitude)")
                        else:
                            result = add_last_seen(db, selected_cdp, location, coords_list)
                            if result:
                                st.success("Last seen record added successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to add last seen - CDP not found")
                    except ValueError:
                        st.error("Invalid coordinates format. Please use numbers only.")
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("No camera detected persons available to add last seen records.")

with tab3:
    st.header("AI Processing")
    
    cdp_list = get_camera_detected_persons(db)
    if cdp_list:
        cdp_ids = [cdp.cameraDetectedPersonId for cdp in cdp_list]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Calculate AI Attributes")
            ai_cdp = st.selectbox("Select CDP for AI Processing", cdp_ids, key="ai_cdp")
            
            if st.button("Calculate AI Attributes", type="primary"):
                with st.spinner("Calculating AI attributes..."):
                    result = calculate_ai_attributes(db, ai_cdp)
                    if result:
                        st.success("AI attributes calculated successfully!")
                        st.write("**Generated AI Attributes:**")
                        st.write(f"- Embedding: {len(result.embedding)} dimensions")
                        st.write(f"- Potentially Lost: {result.potentiallyLost}")
                        st.write(f"- Is Elderly: {result.isElderly}")
                        st.write(f"- Is Disabled: {result.isDisabled}")
                        st.rerun()
                    else:
                        st.error("Failed to calculate AI attributes - CDP not found")
        
        # with col2:
        #     st.subheader("Batch AI Processing")
        #     st.info("Process multiple CDPs at once")
            
        #     unprocessed_cdps = [cdp for cdp in cdp_list if not cdp.embedding or len(cdp.embedding) == 0]
        #     if unprocessed_cdps:
        #         st.write(f"**{len(unprocessed_cdps)}** CDPs without AI attributes")
                
        #         if st.button("Process All Unprocessed CDPs"):
        #             progress_bar = st.progress(0)
        #             success_count = 0
                    
        #             for i, cdp in enumerate(unprocessed_cdps):
        #                 try:
        #                     result = calculate_ai_attributes(db, cdp.cameraDetectedPersonId)
        #                     if result:
        #                         success_count += 1
        #                     progress_bar.progress((i + 1) / len(unprocessed_cdps))
        #                 except Exception as e:
        #                     st.error(f"Error processing CDP {cdp.cameraDetectedPersonId}: {e}")
                    
        #             st.success(f"Batch processing completed! {success_count}/{len(unprocessed_cdps)} CDPs processed successfully")
        #             st.rerun()
        #     else:
        #         st.success("All CDPs have AI attributes calculated!")
    else:
        st.info("No camera detected persons available for AI processing.")

with tab4:
    st.header("Special Actions")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Potentially Lost")
        potentially_lost = get_potentially_lost_persons(db)
        st.metric("Potentially Lost", len(potentially_lost))
        
        if potentially_lost:
            st.write("**Recent Potentially Lost:**")
            for cdp in potentially_lost[:3]:
                st.write(f"- CDP ID: {cdp.cameraDetectedPersonId}")

    with col2:
        st.subheader("Elderly Persons")
        elderly = get_elderly_persons(db)
        st.metric("Elderly Persons", len(elderly))
        
        if elderly:
            st.write("**Recent Elderly:**")
            for cdp in elderly[:3]:
                st.write(f"- CDP ID: {cdp.cameraDetectedPersonId}")

    with col3:
        st.subheader("Disabled Persons")
        disabled = get_disabled_persons(db)
        st.metric("Disabled Persons", len(disabled))
        
        if disabled:
            st.write("**Recent Disabled:**")
            for cdp in disabled[:3]:
                st.write(f"- CDP ID: {cdp.cameraDetectedPersonId}")
    
    # Advanced filtering
    st.subheader("Advanced Filtering")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_lost = st.checkbox("Show Only Potentially Lost", value=False)
    with col2:
        show_elderly = st.checkbox("Show Only Elderly", value=False)
    with col3:
        show_disabled = st.checkbox("Show Only Disabled", value=False)
    
    if st.button("Apply Filters"):
        filtered_cdps = cdp_list
        
        if show_lost:
            filtered_cdps = [cdp for cdp in filtered_cdps if cdp.potentiallyLost]
        if show_elderly:
            filtered_cdps = [cdp for cdp in filtered_cdps if cdp.isElderly]
        if show_disabled:
            filtered_cdps = [cdp for cdp in filtered_cdps if cdp.isDisabled]
        
        if filtered_cdps:
            st.success(f"Found {len(filtered_cdps)} CDPs matching filters")
            filtered_data = []
            for cdp in filtered_cdps:
                filtered_data.append({
                    'ID': cdp.cameraDetectedPersonId,
                    # 'Embedding Length': len(cdp.embedding) if cdp.embedding else 0,
                    'Potentially Lost': 'Yes' if cdp.potentiallyLost else 'No',
                    'Is Elderly': 'Yes' if cdp.isElderly else 'No',
                    'Is Disabled': 'Yes' if cdp.isDisabled else 'No'
                })
            
            df_filtered = pd.DataFrame(filtered_data)
            st.dataframe(df_filtered, use_container_width=True)
        else:
            st.warning("No CDPs match the selected filters")

with tab5:
    st.header("CDP Statistics")
    
    cdp_list = get_camera_detected_persons(db)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cdps = len(cdp_list)
        st.metric("Total CDPs", total_cdps)
    
    # with col2:
    #     with_ai = len([cdp for cdp in cdp_list if cdp.embedding and len(cdp.embedding) > 0])
    #     st.metric("With AI Attributes", with_ai)
    
    # with col3:
    #     without_ai = total_cdps - with_ai
    #     st.metric("Without AI Attributes", without_ai)
    
    with col4:
        with_persons = len([cdp for cdp in cdp_list if cdp.persons])
        st.metric("Linked to Persons", with_persons)
    
    # AI Attributes Statistics
    st.subheader("AI Attributes Distribution")
    
    if cdp_list:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lost_count = len([cdp for cdp in cdp_list if cdp.potentiallyLost])
            st.metric("Potentially Lost", lost_count, 
                     delta=f"{(lost_count/total_cdps*100):.1f}%" if total_cdps > 0 else 0)
        
        with col2:
            elderly_count = len([cdp for cdp in cdp_list if cdp.isElderly])
            st.metric("Elderly", elderly_count,
                     delta=f"{(elderly_count/total_cdps*100):.1f}%" if total_cdps > 0 else 0)
        
        with col3:
            disabled_count = len([cdp for cdp in cdp_list if cdp.isDisabled])
            st.metric("Disabled", disabled_count,
                     delta=f"{(disabled_count/total_cdps*100):.1f}%" if total_cdps > 0 else 0)
    
    # Recent CDPs table
    st.subheader("Recent Camera Detected Persons")
    if cdp_list:
        # Sort by ID descending to get recent ones
        recent_cdps = sorted(cdp_list, key=lambda x: x.cameraDetectedPersonId, reverse=True)[:10]
        
        recent_data = []
        for cdp in recent_cdps:
            recent_data.append({
                'ID': cdp.cameraDetectedPersonId,
                # 'AI Processed': 'Yes' if cdp.embedding and len(cdp.embedding) > 0 else 'No',
                'Potentially Lost': 'Yes' if cdp.potentiallyLost else 'No',
                'Elderly': 'Yes' if cdp.isElderly else 'No',
                'Disabled': 'Yes' if cdp.isDisabled else 'No',
                'Related Persons': len(cdp.persons)
            })
        
        df_recent = pd.DataFrame(recent_data)
        st.dataframe(df_recent, use_container_width=True)
    else:
        st.info("No camera detected person data available.")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:**\n- AI attributes are automatically calculated\n- Use batch processing for multiple CDPs\n- Last seen records help track movement\n- Filter CDPs by AI attributes for analysis")
