import streamlit as st
import pandas as pd
from database import get_db
from crud import *

st.title("Search Data Management")

# Get database session
db = next(get_db())

# Create tabs for different functions
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 All Searches", 
    "➕ Create Search", 
    "🔄 Update Status", 
    "❌ Cancel Search",
    "📊 Statistics"
])

with tab1:
    st.header("Search Data List")
    search_data_list = get_search_data(db)

    if search_data_list:
        data = []
        for search in search_data_list:
            status_icon = "🔄" if search.isProcessing else "✅"
            found_icon = "✅" if search.isFound else "❌"
            
            data.append({
                'Search ID': search.searchID,
                'Request Time': search.requestTime.strftime("%Y-%m-%d %H:%M:%S"),
                'Status': f"{status_icon} {'Processing' if search.isProcessing else 'Completed'}",
                'Found': f"{found_icon} {'Yes' if search.isFound else 'No'}",
                'Requested By': search.requestedBy_id or "Unknown"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Show search details with results
        st.subheader("Search Details with Results")
        search_ids = [search.searchID for search in search_data_list]
        selected_search_id = st.selectbox("Select Search to view details:", search_ids)
        
        if selected_search_id:
            search_data, results_list = get_search_data_with_results(db, selected_search_id)
            if search_data:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Search ID:** {search_data.searchID}")
                    st.write(f"**Request Time:** {search_data.requestTime.strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Status:** {'🔄 Processing' if search_data.isProcessing else '✅ Completed'}")
                    st.write(f"**Found:** {'✅ Yes' if search_data.isFound else '❌ No'}")
                
                with col2:
                    staff_name = "Unknown"
                    if search_data.requestedBy:
                        staff_name = search_data.requestedBy.name
                    st.write(f"**Requested By:** {staff_name} (ID: {search_data.requestedBy_id})")
                    st.write(f"**Results Count:** {len(results_list)}")
                
                # Show results
                if results_list:
                    st.write("**Search Results:**")
                    results_data = []
                    for result in results_list:
                        status_icon = "✅" if result.isAccepted else "❌"
                        results_data.append({
                            'Result ID': result.id,
                            'CDP ID': result.cameraDetectedPersonId,
                            'Accepted': f"{status_icon} {'Yes' if result.isAccepted else 'No'}"
                        })
                    
                    df_results = pd.DataFrame(results_data)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
                else:
                    st.info("No results found for this search")
    else:
        st.info("No search data available.")

with tab2:
    st.header("Create New Search")
    
    staff_list = get_security_staffs(db)
    
    if staff_list:
        with st.form("create_search_form"):
            # Create staff options with names
            staff_options = {staff.staffId: f"{staff.name} (ID: {staff.staffId})" for staff in staff_list}
            
            requested_by = st.selectbox("Requested By Staff*", 
                                      options=list(staff_options.keys()),
                                      format_func=lambda x: staff_options[x],
                                      help="Select the security staff member requesting the search")
            
            submitted = st.form_submit_button("Create Search Request")
            if submitted:
                try:
                    new_search = create_search_data(db, requested_by)
                    st.success(f"✅ Search created successfully!")
                    st.info(f"**Search ID:** {new_search.searchID}")
                    st.info(f"**Request Time:** {new_search.requestTime.strftime('%Y-%m-%d %H:%M:%S')}")
                    st.info(f"**Status:** Processing")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    else:
        st.error("No security staff available. Please add security staff first.")

with tab3:
    st.header("Update Search Status")
    
    search_data_list = get_search_data(db)
    if search_data_list:
        # Filter only processing searches for status update
        processing_searches = [search for search in search_data_list if search.isProcessing]
        
        if processing_searches:
            search_options = {search.searchID: f"Search {search.searchID} (Requested by: {search.requestedBy_id})" 
                            for search in processing_searches}
            
            selected_search = st.selectbox("Select Search to Update:", 
                                         options=list(search_options.keys()),
                                         format_func=lambda x: search_options[x])
            
            new_status = st.radio("Search Result:", 
                                [True, False], 
                                format_func=lambda x: "✅ Found - Mark as successful" if x else "❌ Not Found - Mark as completed but unsuccessful",
                                help="Update the search status and mark as completed")
            
            if st.button("Update Search Status", type="primary"):
                with st.spinner("Updating search status..."):
                    result = update_search_status(db, selected_search, new_status)
                    if result:
                        st.success("✅ Search status updated successfully!")
                        st.info(f"Search {selected_search} marked as {'found' if new_status else 'not found'} and completed")
                        st.rerun()
                    else:
                        st.error("❌ Failed to update search status")
        else:
            st.info("No processing searches available for status update")
            
        # Show all searches for reference
        st.subheader("All Searches Reference")
        if search_data_list:
            ref_data = []
            for search in search_data_list:
                status_icon = "🔄" if search.isProcessing else "✅"
                found_icon = "✅" if search.isFound else "❌"
                
                ref_data.append({
                    'Search ID': search.searchID,
                    'Status': f"{status_icon} {'Processing' if search.isProcessing else 'Completed'}",
                    'Found': f"{found_icon} {'Yes' if search.isFound else 'No'}",
                    'Requested By': search.requestedBy_id or "Unknown"
                })
            
            df_ref = pd.DataFrame(ref_data)
            st.dataframe(df_ref, use_container_width=True, hide_index=True)
    else:
        st.info("No search data available")

with tab4:
    st.header("Cancel Search")
    
    search_data_list = get_search_data(db)
    if search_data_list:
        # Only show processing searches for cancellation
        cancellable_searches = [search for search in search_data_list if search.isProcessing]
        
        if cancellable_searches:
            cancel_options = {search.searchID: f"Search {search.searchID} (Requested: {search.requestTime.strftime('%Y-%m-%d %H:%M')})" 
                            for search in cancellable_searches}
            
            cancel_search_id = st.selectbox("Select Search to Cancel:", 
                                          options=list(cancel_options.keys()),
                                          format_func=lambda x: cancel_options[x])
            
            # Show search details before cancellation
            if cancel_search_id:
                search_to_cancel = get_search_data_by_id(db, cancel_search_id)
                if search_to_cancel:
                    st.warning("⚠️ Search Details:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Search ID:** {search_to_cancel.searchID}")
                        st.write(f"**Request Time:** {search_to_cancel.requestTime.strftime('%Y-%m-%d %H:%M:%S')}")
                    with col2:
                        st.write(f"**Status:** Processing")
                        st.write(f"**Requested By:** {search_to_cancel.requestedBy_id}")
            
            # Confirmation
            st.error("🚨 Danger Zone")
            confirmation = st.text_input("Type 'CANCEL' to confirm cancellation:")
            
            if st.button("Cancel Search", type="primary"):
                if confirmation == "CANCEL":
                    with st.spinner("Cancelling search..."):
                        result = cancel_search(db, cancel_search_id)
                        if "error" in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            st.success("✅ " + result["message"])
                            st.rerun()
                else:
                    st.error("❌ Please type 'CANCEL' to confirm cancellation")
        else:
            st.info("No processing searches available for cancellation")
            
        # Show completed searches for reference
        st.subheader("Completed Searches (Not Cancellable)")
        completed_searches = [search for search in search_data_list if not search.isProcessing]
        if completed_searches:
            completed_data = []
            for search in completed_searches:
                found_icon = "✅" if search.isFound else "❌"
                completed_data.append({
                    'Search ID': search.searchID,
                    'Found': f"{found_icon} {'Yes' if search.isFound else 'No'}",
                    'Request Time': search.requestTime.strftime('%Y-%m-%d %H:%M'),
                    'Requested By': search.requestedBy_id or "Unknown"
                })
            
            df_completed = pd.DataFrame(completed_data)
            st.dataframe(df_completed, use_container_width=True, hide_index=True)
    else:
        st.info("No search data available")

with tab5:
    st.header("Search Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_searches = get_search_data_count(db)
        st.metric("Total Searches", total_searches)
    
    with col2:
        active_searches = len(get_active_searches(db))
        st.metric("Active Searches", active_searches)
    
    with col3:
        completed_searches = len(get_completed_searches(db))
        st.metric("Completed Searches", completed_searches)
    
    with col4:
        successful_searches = len(get_successful_searches(db))
        st.metric("Successful Searches", successful_searches)
    
    # Success rate
    st.subheader("Success Rate")
    if total_searches > 0:
        success_rate = (successful_searches / total_searches) * 100
        completion_rate = (completed_searches / total_searches) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col2:
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
        # Progress bars
        st.write("**Progress Overview:**")
        st.progress(success_rate/100, text=f"Successful: {success_rate:.1f}%")
        st.progress(completion_rate/100, text=f"Completed: {completion_rate:.1f}%")
    
    # Recent searches table
    st.subheader("Recent Search Activity")
    search_data_list = get_search_data(db)
    if search_data_list:
        # Sort by search ID descending to get recent ones
        recent_searches = sorted(search_data_list, key=lambda x: x.searchID, reverse=True)[:10]
        
        recent_data = []
        for search in recent_searches:
            status_icon = "🔄" if search.isProcessing else "✅"
            found_icon = "✅" if search.isFound else "❌"
            
            recent_data.append({
                'Search ID': search.searchID,
                'Time': search.requestTime.strftime('%Y-%m-%d %H:%M'),
                'Status': f"{status_icon} {'Processing' if search.isProcessing else 'Completed'}",
                'Result': f"{found_icon} {'Found' if search.isFound else 'Not Found'}",
                'Staff ID': search.requestedBy_id or "Unknown"
            })
        
        df_recent = pd.DataFrame(recent_data)
        st.dataframe(df_recent, use_container_width=True, hide_index=True)
    else:
        st.info("No search data available")
    
    # Staff performance (if we have data)
    st.subheader("Staff Search Performance")
    staff_list = get_security_staffs(db)
    if staff_list and search_data_list:
        staff_performance = []
        for staff in staff_list:
            staff_searches = [s for s in search_data_list if s.requestedBy_id == staff.staffId]
            if staff_searches:
                total_staff_searches = len(staff_searches)
                successful_staff_searches = len([s for s in staff_searches if s.isFound])
                success_rate_staff = (successful_staff_searches / total_staff_searches * 100) if total_staff_searches > 0 else 0
                
                staff_performance.append({
                    'Staff Name': staff.name,
                    'Total Searches': total_staff_searches,
                    'Successful': successful_staff_searches,
                    'Success Rate': f"{success_rate_staff:.1f}%"
                })
        
        if staff_performance:
            df_staff = pd.DataFrame(staff_performance)
            st.dataframe(df_staff, use_container_width=True, hide_index=True)
        else:
            st.info("No staff performance data available")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:**\n- Only processing searches can be updated or cancelled\n- Mark searches as found when persons are located\n- Use cancellation for abandoned search requests\n- Check statistics for performance insights")
