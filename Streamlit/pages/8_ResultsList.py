import streamlit as st
import pandas as pd
from database import get_db
from crud import *

st.title("Results List Management")

# Get database session
db = next(get_db())

# Create tabs for different functions
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 All Results", 
    "➕ Add Result", 
    "🔍 Search Results", 
    "✏️ Update Results",
    "📊 Statistics"
])

with tab1:
    st.header("Results List")
    results_list = get_results_list(db)

    if results_list:
        data = []
        for result in results_list:
            status_icon = "✅" if result.isAccepted else "❌"
            status_text = "Accepted" if result.isAccepted else "Rejected"
            
            data.append({
                'Result ID': result.id,
                'Search ID': result.searchID,
                'CDP ID': result.cameraDetectedPersonId,
                'Status': f"{status_icon} {status_text}"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Show result details
        st.subheader("Result Details")
        result_ids = [result.id for result in results_list]
        selected_result_id = st.selectbox("Select Result to view details:", result_ids)
        
        if selected_result_id:
            result = get_results_list_by_id(db, selected_result_id)
            if result:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Result ID:** {result.id}")
                    st.write(f"**Search ID:** {result.searchID}")
                    st.write(f"**CDP ID:** {result.cameraDetectedPersonId}")
                
                with col2:
                    status_icon = "✅" if result.isAccepted else "❌"
                    st.write(f"**Status:** {status_icon} {'Accepted' if result.isAccepted else 'Rejected'}")
                
                # Show related search information
                search_data = get_search_data_by_id(db, result.searchID)
                if search_data:
                    st.write("**Search Information:**")
                    st.write(f"- Request Time: {search_data.requestTime.strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"- Search Status: {'Processing' if search_data.isProcessing else 'Completed'}")
                    st.write(f"- Found: {'Yes' if search_data.isFound else 'No'}")
                
                # Show related CDP information
                cdp = get_camera_detected_person_by_id(db, result.cameraDetectedPersonId)
                if cdp:
                    st.write("**Camera Detected Person:**")
                    st.write(f"- Potentially Lost: {'Yes' if cdp.potentiallyLost else 'No'}")
                    st.write(f"- Elderly: {'Yes' if cdp.isElderly else 'No'}")
                    st.write(f"- Disabled: {'Yes' if cdp.isDisabled else 'No'}")
    else:
        st.info("No results data available.")

with tab2:
    st.header("Add New Result")
    
    search_list = get_search_data(db)
    cdp_list = get_camera_detected_persons(db)
    
    if search_list and cdp_list:
        with st.form("add_result_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Search options
                search_options = {search.searchID: f"Search {search.searchID} (Requested by: {search.requestedBy_id})" 
                                for search in search_list}
                search_id = st.selectbox("Search ID*", 
                                       options=list(search_options.keys()),
                                       format_func=lambda x: search_options[x],
                                       help="Select the search this result belongs to")
            
            with col2:
                # CDP options
                cdp_options = {cdp.cameraDetectedPersonId: f"CDP {cdp.cameraDetectedPersonId}" 
                              for cdp in cdp_list}
                cdp_id = st.selectbox("Camera Detected Person ID*", 
                                    options=list(cdp_options.keys()),
                                    format_func=lambda x: cdp_options[x],
                                    help="Select the camera detected person for this result")
            
            # Status selection
            is_accepted = st.radio("Result Status*", 
                                 [True, False], 
                                 format_func=lambda x: "✅ Accepted - Match confirmed" if x else "❌ Rejected - Not a match",
                                 horizontal=True,
                                 help="Accept if this is a correct match, reject if it's not")
            
            submitted = st.form_submit_button("Add Result")
            if submitted:
                try:
                    # Check if this combination already exists
                    existing_results = get_results_list(db)
                    duplicate = any(r for r in existing_results if r.searchID == search_id and r.cameraDetectedPersonId == cdp_id)
                    
                    if duplicate:
                        st.error("❌ This search-CDP combination already exists!")
                    else:
                        new_result = create_results_list(db, search_id, cdp_id, is_accepted)
                        st.success("✅ Result added successfully!")
                        st.info(f"**Result ID:** {new_result.id}")
                        st.info(f"**Search ID:** {new_result.searchID}")
                        st.info(f"**CDP ID:** {new_result.cameraDetectedPersonId}")
                        st.info(f"**Status:** {'✅ Accepted' if new_result.isAccepted else '❌ Rejected'}")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    else:
        if not search_list:
            st.error("No search data available. Please create searches first.")
        if not cdp_list:
            st.error("No camera detected persons available. Please add CDPs first.")

with tab3:
    st.header("Search Specific Results")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Results by Search")
        search_list = get_search_data(db)
        if search_list:
            search_options = {search.searchID: f"Search {search.searchID}" for search in search_list}
            selected_search = st.selectbox("Select Search:", 
                                         options=list(search_options.keys()),
                                         format_func=lambda x: search_options[x])
            
            if st.button("Get Search Results", key="search_btn"):
                with st.spinner("Fetching results..."):
                    results = get_results_by_search(db, selected_search)
                    if results:
                        st.success(f"✅ Found {len(results)} results for search {selected_search}")
                        
                        search_data = []
                        for result in results:
                            status_icon = "✅" if result.isAccepted else "❌"
                            search_data.append({
                                'Result ID': result.id,
                                'CDP ID': result.cameraDetectedPersonId,
                                'Status': f"{status_icon} {'Accepted' if result.isAccepted else 'Rejected'}"
                            })
                        
                        df_search = pd.DataFrame(search_data)
                        st.dataframe(df_search, use_container_width=True, hide_index=True)
                        
                        # Statistics for this search
                        accepted_count = len([r for r in results if r.isAccepted])
                        acceptance_rate = (accepted_count / len(results) * 100) if results else 0
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Results", len(results))
                        with col2:
                            st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")
                    else:
                        st.warning(f"⚠️ No results found for search {selected_search}")
        else:
            st.info("No search data available")

    with col2:
        st.subheader("Results by CDP")
        cdp_list = get_camera_detected_persons(db)
        if cdp_list:
            cdp_options = {cdp.cameraDetectedPersonId: f"CDP {cdp.cameraDetectedPersonId}" for cdp in cdp_list}
            selected_cdp = st.selectbox("Select CDP:", 
                                      options=list(cdp_options.keys()),
                                      format_func=lambda x: cdp_options[x])
            
            if st.button("Get CDP Results", key="cdp_btn"):
                with st.spinner("Fetching results..."):
                    results = get_results_by_person(db, selected_cdp)
                    if results:
                        st.success(f"✅ Found {len(results)} results for CDP {selected_cdp}")
                        
                        cdp_data = []
                        for result in results:
                            status_icon = "✅" if result.isAccepted else "❌"
                            cdp_data.append({
                                'Result ID': result.id,
                                'Search ID': result.searchID,
                                'Status': f"{status_icon} {'Accepted' if result.isAccepted else 'Rejected'}"
                            })
                        
                        df_cdp = pd.DataFrame(cdp_data)
                        st.dataframe(df_cdp, use_container_width=True, hide_index=True)
                        
                        # Statistics for this CDP
                        accepted_count = len([r for r in results if r.isAccepted])
                        acceptance_rate = (accepted_count / len(results) * 100) if results else 0
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Results", len(results))
                        with col2:
                            st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")
                    else:
                        st.warning(f"⚠️ No results found for CDP {selected_cdp}")
        else:
            st.info("No camera detected persons available")

with tab4:
    st.header("Update Results")
    
    results_list = get_results_list(db)
    if results_list:
        result_options = {result.id: f"Result {result.id} (Search: {result.searchID}, CDP: {result.cameraDetectedPersonId})" 
                        for result in results_list}
        
        selected_result = st.selectbox("Select Result to Update:", 
                                     options=list(result_options.keys()),
                                     format_func=lambda x: result_options[x])
        
        if selected_result:
            result = get_results_list_by_id(db, selected_result)
            if result:
                st.write("### Current Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_input("Result ID", value=result.id, disabled=True)
                    st.text_input("Search ID", value=result.searchID, disabled=True)
                with col2:
                    st.text_input("CDP ID", value=result.cameraDetectedPersonId, disabled=True)
                    current_status = "✅ Accepted" if result.isAccepted else "❌ Rejected"
                    st.text_input("Current Status", value=current_status, disabled=True)
                
                # Update form
                with st.form("update_result_form"):
                    new_status = st.radio("New Status", 
                                        [True, False], 
                                        format_func=lambda x: "✅ Accept" if x else "❌ Reject",
                                        index=0 if result.isAccepted else 1,
                                        horizontal=True)
                    
                    submitted = st.form_submit_button("Update Result")
                    if submitted:
                        try:
                            updated_result = update_results_list(db, selected_result, isAccepted=new_status)
                            if updated_result:
                                st.success("✅ Result updated successfully!")
                                st.info(f"**New Status:** {'✅ Accepted' if updated_result.isAccepted else '❌ Rejected'}")
                                st.rerun()
                            else:
                                st.error("❌ Failed to update result")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
    else:
        st.info("No results available to update")

with tab5:
    st.header("Results Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_results = get_results_list_count(db)
        st.metric("Total Results", total_results)
    
    with col2:
        accepted_results = len(get_accepted_results(db))
        st.metric("Accepted", accepted_results)
    
    with col3:
        rejected_results = len(get_rejected_results(db))
        st.metric("Rejected", rejected_results)
    
    with col4:
        acceptance_rate = (accepted_results / total_results * 100) if total_results > 0 else 0
        st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")
    
    # Visual progress bars
    st.subheader("Distribution")
    if total_results > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Acceptance Distribution:**")
            accepted_percent = (accepted_results / total_results) * 100
            rejected_percent = (rejected_results / total_results) * 100
            
            st.progress(accepted_percent/100, text=f"Accepted: {accepted_percent:.1f}%")
            st.progress(rejected_percent/100, text=f"Rejected: {rejected_percent:.1f}%")
        
        with col2:
            st.write("**Quick Stats:**")
            st.write(f"✅ **Accepted:** {accepted_results} results")
            st.write(f"❌ **Rejected:** {rejected_results} results")
            st.write(f"📊 **Total:** {total_results} results")
    
    # Recent activity
    st.subheader("Recent Results Activity")
    results_list = get_results_list(db)
    if results_list:
        # Sort by ID descending to get recent ones
        recent_results = sorted(results_list, key=lambda x: x.id, reverse=True)[:10]
        
        recent_data = []
        for result in recent_results:
            status_icon = "✅" if result.isAccepted else "❌"
            recent_data.append({
                'Result ID': result.id,
                'Search ID': result.searchID,
                'CDP ID': result.cameraDetectedPersonId,
                'Status': f"{status_icon} {'Accepted' if result.isAccepted else 'Rejected'}"
            })
        
        df_recent = pd.DataFrame(recent_data)
        st.dataframe(df_recent, use_container_width=True, hide_index=True)
    else:
        st.info("No results data available")
    
    # Search performance
    st.subheader("Search Performance")
    search_list = get_search_data(db)
    if search_list and results_list:
        search_performance = []
        for search in search_list:
            search_results = get_results_by_search(db, search.searchID)
            if search_results:
                total_search_results = len(search_results)
                accepted_search_results = len([r for r in search_results if r.isAccepted])
                acceptance_rate_search = (accepted_search_results / total_search_results * 100) if total_search_results > 0 else 0
                
                search_performance.append({
                    'Search ID': search.searchID,
                    'Total Results': total_search_results,
                    'Accepted': accepted_search_results,
                    'Acceptance Rate': f"{acceptance_rate_search:.1f}%",
                    'Search Status': 'Processing' if search.isProcessing else 'Completed'
                })
        
        if search_performance:
            df_performance = pd.DataFrame(search_performance)
            st.dataframe(df_performance, use_container_width=True, hide_index=True)
        else:
            st.info("No search performance data available")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:**\n- Accept results when matches are confirmed\n- Reject incorrect matches to improve accuracy\n- Check acceptance rates for search quality\n- Update results status as new information arrives")
