import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from Embedding_worker.helpers import extract_frame_by_count, get_person_crop_from_coords

st.set_page_config(page_title="Search Results", layout="wide")
st.title("🔍 Search Results")

# Get the path to search_results.json
json_file_path = Path(__file__).parent.parent.parent / "search_results.json"

# Load search results from JSON
@st.cache_data
def load_search_results():
    if json_file_path.exists():
        with open(json_file_path, 'r') as f:
            return json.load(f)
    return {}

try:
    search_results = load_search_results()
    available_ids = sorted([int(key) for key in search_results.keys()])
    
    if not available_ids:
        st.warning("No search results found in search_results.json")
    else:
        # Search interface
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_id = st.selectbox(
                "🔍 Search for a person by ID",
                options=available_ids,
                format_func=lambda x: f"Person ID: {x}"
            )
        with col2:
            st.write("")  # Spacing
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        # Display selected search result
        if selected_id:
            result = search_results.get(str(selected_id))
            
            if result:
                # Summary section
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Person ID", result.get("person_id", "N/A"))
                with col2:
                    st.metric("Total Matches", result.get("total_matches", 0))
                with col3:
                    matches = result.get("matches", [])
                    st.metric("Displayed Results", len(matches))
                
                st.divider()
                
                # Matches display
                matches = result.get("matches", [])
                
                if matches:
                    st.subheader(f"📊 Showing {len(matches)} Matches")
                    
                    # Create a DataFrame for better visualization
                    saved_paths = st.session_state.saved_paths
                    vid_location_mapping = {idx:path for idx,path in enumerate(saved_paths)}
                    data = []
                    for match in matches:
                        data.append({
                            "Rank": match.get("rank", "N/A"),
                            "Detected Person ID": match.get("detected_person_id", "N/A"),
                            "Confidence Score": f"{match.get('confidence_score', 0):.4f}",
                        })
                    
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Detailed view for each match - no expanders
                    st.subheader("📋 Detailed View")
                    
                    for idx, match in enumerate(matches[:10], 1):  # Show top 10 in detail
                        coords = match.get("coords")
                        frame_count = match.get("frame_count")
                        cam_id = match.get("cam_id")
                        video_path = vid_location_mapping.get(cam_id)
                        
                        st.markdown(f"### Rank #{match.get('rank', idx)} - Detected Person: {match.get('detected_person_id', 'N/A')} (Confidence: {match.get('confidence_score', 0):.4f})")
                        
                        if not video_path:
                            st.warning(f"Video path not found for camera {cam_id}")
                            continue
                        
                        frame, frame_w, frame_h = extract_frame_by_count(video_path, frame_count)
                        if frame is None:
                            st.warning(f"Could not extract frame at count {frame_count}")
                            continue
                        
                        crop_image = get_person_crop_from_coords(frame, coords, frame_w, frame_h)
                        if crop_image is None:
                            st.warning(f"Could not crop image at coordinates {coords}")
                            continue

                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            st.image(crop_image, use_column_width=True)
                        
                        with col2:
                            st.write(f"**Rank:** {match.get('rank', 'N/A')}")
                            st.write(f"**Detected Person ID:** {match.get('detected_person_id', 'N/A')}")
                            st.write(f"**Confidence Score:** {match.get('confidence_score', 0):.6f}")
                            st.write(f"**Camera ID:** {match.get('cam_id', 'N/A')}")
                        
                        with col3:
                            st.write(f"**Frame Count:** {match.get('frame_count', 'N/A')}")
                            st.write(f"**Coordinates (x1, y1, x2, y2):** {match.get('coords', [])}")
                            st.write(f"**Is Lost:** {'Yes' if match.get('is_lost') else ('No' if match.get('is_lost') is False else 'N/A')}")
                            st.write(f"**Is Elderly:** {'Yes' if match.get('is_elderly') else ('No' if match.get('is_elderly') is False else 'N/A')}")
                            st.write(f"**Is Disabled:** {'Yes' if match.get('is_disabled') else ('No' if match.get('is_disabled') is False else 'N/A')}")
                        
                        st.divider()
                    
                    if len(matches) > 10:
                        st.info(f"Showing top 10 of {len(matches)} matches. Scroll through the table above to see all results.")
                else:
                    st.info("No matches found for this search result.")
            else:
                st.error(f"Result not found for Person ID: {selected_id}")

except Exception as e:
    st.error(f"Error loading search results: {str(e)}")
    st.error("Make sure search_results.json exists in the project root directory.")