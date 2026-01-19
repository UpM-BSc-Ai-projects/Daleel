import streamlit as st
import pandas as pd
from database import get_db
from crud import *

st.title("Last Seen Management")

# Get database session
db = next(get_db())

# Create tabs for different functions
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 All Records", 
    "🕒 Recent Sightings", 
    "🔍 Search Location", 
    "➕ Add Record",
    "📊 Statistics"
])

with tab1:
    st.header("Last Seen Records")
    last_seen_list = get_last_seen_records(db)

    if last_seen_list:
        data = []
        for record in last_seen_list:
            data.append({
                'ID': record.id,
                'CDP ID': record.CDPid,
                'Location': record.location,
                'Time': record.time.strftime("%Y-%m-%d %H:%M:%S"),
                'Coordinates': str(record.coordinates)
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Show record details
        st.subheader("Record Details")
        record_ids = [record.id for record in last_seen_list]
        selected_record_id = st.selectbox("Select Record to view details:", record_ids)
        
        if selected_record_id:
            record = get_last_seen_by_id(db, selected_record_id)
            if record:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Record ID:** {record.id}")
                    st.write(f"**CDP ID:** {record.CDPid}")
                    st.write(f"**Location:** {record.location}")
                
                with col2:
                    st.write(f"**Time:** {record.time.strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Coordinates:** {record.coordinates}")
                
                # Show related CDP information if available
                cdp = get_camera_detected_person_by_id(db, record.CDPid)
                if cdp:
                    st.write("**Related Camera Detected Person:**")
                    st.write(f"- Potentially Lost: {'Yes' if cdp.potentiallyLost else 'No'}")
                    st.write(f"- Elderly: {'Yes' if cdp.isElderly else 'No'}")
                    st.write(f"- Disabled: {'Yes' if cdp.isDisabled else 'No'}")
    else:
        st.info("No last seen records available.")

with tab2:
    st.header("Recent Sightings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        hours = st.slider("Hours to look back", 1, 168, 24,
                         help="Select how many hours to look back for sightings")
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("Refresh", key="refresh_recent"):
            st.rerun()
    
    recent_sightings = get_recent_sightings(db, hours)

    if recent_sightings:
        st.metric(f"Sightings in last {hours} hours", len(recent_sightings))
        
        # Sort by time descending
        recent_sightings_sorted = sorted(recent_sightings, key=lambda x: x.time, reverse=True)
        
        recent_data = []
        for sighting in recent_sightings_sorted:
            recent_data.append({
                'Record ID': sighting.id,
                'CDP ID': sighting.CDPid,
                'Location': sighting.location,
                'Time': sighting.time.strftime("%Y-%m-%d %H:%M:%S"),
                'Coordinates': str(sighting.coordinates)
            })
        
        df_recent = pd.DataFrame(recent_data)
        st.dataframe(df_recent, use_container_width=True)
        
        # Show on map if coordinates available
        st.subheader("Recent Sightings Map")
        try:
            map_data = []
            for sighting in recent_sightings_sorted[:20]:  # Limit to 20 for performance
                if sighting.coordinates and len(sighting.coordinates) == 2:
                    map_data.append({
                        'lat': sighting.coordinates[0],
                        'lon': sighting.coordinates[1],
                        'location': sighting.location,
                        'cdp_id': sighting.CDPid
                    })
            
            if map_data:
                df_map = pd.DataFrame(map_data)
                st.map(df_map, use_container_width=True)
            else:
                st.info("No valid coordinates available for mapping")
        except Exception as e:
            st.warning(f"Could not display map: {e}")
    else:
        st.info(f"No sightings in the last {hours} hours")

with tab3:
    st.header("Search by Location")
    
    with st.form("location_search_form"):
        location_search = st.text_input("Enter location to search*", 
                                       help="Search for sightings in specific locations")
        search_submitted = st.form_submit_button("Search Location")
    
    if search_submitted:
        if location_search:
            with st.spinner("Searching..."):
                results = get_sightings_by_location(db, location_search)
                
            if results:
                st.success(f"Found {len(results)} records for location: '{location_search}'")
                
                location_data = []
                for result in results:
                    location_data.append({
                        'Record ID': result.id,
                        'CDP ID': result.CDPid,
                        'Location': result.location,
                        'Time': result.time.strftime("%Y-%m-%d %H:%M:%S"),
                        'Coordinates': str(result.coordinates)
                    })
                
                df_location = pd.DataFrame(location_data)
                st.dataframe(df_location, use_container_width=True)
                
                # Show statistics for this location
                col1, col2, col3 = st.columns(3)
                with col1:
                    unique_cdps = len(set(result.CDPid for result in results))
                    st.metric("Unique CDPs", unique_cdps)
                
                with col2:
                    latest_time = max(result.time for result in results)
                    st.metric("Latest Sighting", latest_time.strftime("%H:%M"))
                
                with col3:
                    location_count = len(results)
                    st.metric("Total Sightings", location_count)
            else:
                st.warning(f"No records found for location: '{location_search}'")
        else:
            st.error("Please enter a location to search")

with tab4:
    st.header("Add New Last Seen Record")
    
    # Get available CDPs
    cdp_list = get_camera_detected_persons(db)
    
    if cdp_list:
        with st.form("add_last_seen_record_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                cdp_options = {cdp.cameraDetectedPersonId: f"CDP {cdp.cameraDetectedPersonId}" for cdp in cdp_list}
                selected_cdp = st.selectbox("Camera Detected Person*", 
                                          options=list(cdp_options.keys()),
                                          format_func=lambda x: cdp_options[x])
                
                location = st.text_input("Location*", value="Camera XXX", max_chars=255,
                                       help="Maximum 255 characters")
            
            with col2:
                coordinates_input = st.text_input("Coordinates (latitude,longitude)*", 
                                                value="24.7136,46.6753",
                                                help="Enter as 'latitude,longitude'. Example: 24.7136,46.6753 for Riyadh")
                
                custom_time = st.checkbox("Use custom time")
                if custom_time:
                    custom_datetime = st.datetime_input("Custom Date/Time")
                else:
                    st.info("Current time will be used automatically")
            
            submitted = st.form_submit_button("Add Last Seen Record")
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
                            # Validate coordinate ranges
                            lat, lon = coords_list
                            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                                st.error("Invalid coordinates: latitude must be between -90 and 90, longitude between -180 and 180")
                            else:
                                time_to_use = custom_datetime if custom_time else None
                                result = create_last_seen(db, selected_cdp, location, coords_list, time_to_use)
                                if result:
                                    st.success("Last seen record added successfully!")
                                    st.rerun()
                                else:
                                    st.error("Failed to add last seen record")
                    except ValueError:
                        st.error("Invalid coordinates format. Please use numbers only (e.g., 24.7136,46.6753)")
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("No camera detected persons available. Please add CDPs first.")

with tab5:
    st.header("Last Seen Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = get_last_seen_count(db)
        st.metric("Total Records", total_records)
    
    with col2:
        last_24h = len(get_recent_sightings(db, 24))
        st.metric("Last 24 Hours", last_24h)
    
    with col3:
        last_week = len(get_recent_sightings(db, 168))  # 168 hours = 7 days
        st.metric("Last 7 Days", last_week)
    
    with col4:
        unique_locations = len(set(record.location for record in get_last_seen_records(db)))
        st.metric("Unique Locations", unique_locations)
    
    # Time-based statistics
    st.subheader("Time Distribution")
    
    last_seen_list = get_last_seen_records(db)
    if last_seen_list:
        # Group by hour of day
        hours_count = {}
        for record in last_seen_list:
            hour = record.time.hour
            hours_count[hour] = hours_count.get(hour, 0) + 1
        
        # Create hour distribution data
        hour_data = []
        for hour in range(24):
            count = hours_count.get(hour, 0)
            hour_data.append({
                'Hour': f"{hour:02d}:00",
                'Count': count,
                'Percentage': f"{(count/len(last_seen_list)*100):.1f}%" if last_seen_list else "0%"
            })
        
        df_hours = pd.DataFrame(hour_data)
        st.dataframe(df_hours, use_container_width=True)
        
        # Location frequency
        st.subheader("Top Locations")
        location_count = {}
        for record in last_seen_list:
            location = record.location
            location_count[location] = location_count.get(location, 0) + 1
        
        # Get top 10 locations
        top_locations = sorted(location_count.items(), key=lambda x: x[1], reverse=True)[:10]
        
        location_data = []
        for location, count in top_locations:
            location_data.append({
                'Location': location,
                'Sightings': count,
                'Percentage': f"{(count/len(last_seen_list)*100):.1f}%"
            })
        
        df_locations = pd.DataFrame(location_data)
        st.dataframe(df_locations, use_container_width=True)
    
    # Recent activity
    st.subheader("Recent Activity")
    recent_records = get_last_seen_records(db)
    if recent_records:
        # Sort by time descending and take top 10
        recent_records_sorted = sorted(recent_records, key=lambda x: x.time, reverse=True)[:10]
        
        recent_activity_data = []
        for record in recent_records_sorted:
            recent_activity_data.append({
                'Time': record.time.strftime("%Y-%m-%d %H:%M:%S"),
                'Location': record.location,
                'CDP ID': record.CDPid,
                'Coordinates': f"{record.coordinates[0]:.4f}, {record.coordinates[1]:.4f}" if record.coordinates else "N/A"
            })
        
        df_recent_activity = pd.DataFrame(recent_activity_data)
        st.dataframe(df_recent_activity, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:**\n- Use the map to visualize recent sightings\n- Search by location to find patterns\n- Add custom time for historical records\n- Check statistics for time and location patterns")
