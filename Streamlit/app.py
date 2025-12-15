import streamlit as st
from models import create_tables,initialize_collections
import tempfile
from pathlib import Path
import os
import time
import multiprocessing as mp
from multiprocessing import get_context
from typing import List
from MCDPT.multiprocessing_mct import process_stream
from Embedding_worker.embedding_worker import *
from database import SessionLocal  # import your session creator
from crud import get_persons_count, get_lost_persons



st.set_page_config(
    page_title="Lost Persons Management System",
    page_icon="",
    layout="wide"
)

# ============================================================
# ONE-TIME INITIALIZATION (using session state guards)
# ============================================================

def ensure_session_state():
    """Initialize all session state variables on first run."""
    if "processes" not in st.session_state:
        st.session_state.processes = {}  # id -> mp.Process
    if "process_meta" not in st.session_state:
        st.session_state.process_meta = {}  # id -> metadata dict
    if "mp_ctx" not in st.session_state:
        st.session_state.mp_ctx = get_context("spawn")
        st.session_state.lock = st.session_state.mp_ctx.Lock()
        st.session_state.db_lock = st.session_state.mp_ctx.Lock()
        st.session_state.Pid = st.session_state.mp_ctx.Value("i", 0)
    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False
    if "collections_initialized" not in st.session_state:
        st.session_state.collections_initialized = False

# Initialize session state first thing
ensure_session_state()

# Create database tables (only once)
if not st.session_state.db_initialized:
    create_tables()
    st.session_state.db_initialized = True

# Initialize Qdrant collections (only once)
if not st.session_state.collections_initialized:
    initialize_collections()
    st.session_state.collections_initialized = True


st.title("Lost Persons Management System")
st.markdown("""
### Welcome to the Integrated Lost Persons Management System

Use the sidebar to navigate between sections:

- **Admins** - Manage administrator accounts
- **Security Staff** - Manage security personnel  
- **Family Members** - Manage families of missing persons
- **Persons** - Manage missing and found persons
- **Camera Detected Persons** - Automatically detected persons
- **Last Seen** - Last known locations of persons
- **Search Data** - Manage search requests
- **Results List** - Search results and matches

---
""")





# -----------------------------
# Utilities
# -----------------------------

def save_uploaded_files(uploaded_files) -> List[str]:
    """Write uploaded files to temporary files and return a list of paths."""
    paths = []
    for f in uploaded_files:
        suffix = Path(f.name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(f.read())
        tmp.flush()
        tmp.close()
        paths.append(tmp.name)
    return paths

def check_lost_persons():
    """Check for lost persons in the database."""
    db = SessionLocal()
    lost_persons = get_lost_persons(db)
    if lost_persons:
        all_person_results = lost_person_search(lost_persons=lost_persons)
        return all_person_results
    
    return lost_persons




# -----------------------------
# Streamlit UI
# -----------------------------




col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Upload video(s)", type=["mp4", "mov", "avi"], accept_multiple_files=True
    )

    cam_options = list(range(0, 6))  # default options to choose from; adjust as needed
    selected_cams = st.multiselect("Select webcam IDs to open", cam_options, default=[])

    start_button = st.button("Start Selected")
    stop_button = st.button("Stop All")

with col2:
    st.write("### Running processes")
    if st.session_state.processes:
        for pid_key, proc in st.session_state.processes.items():
            meta = st.session_state.process_meta.get(pid_key, {})
            status = "alive" if proc.is_alive() else "stopped"
            st.write(f"**{pid_key}** — {meta.get('type')} — {meta.get('input')} — {status}")
    else:
        st.write("No active processes")


# Create multiprocessing context and shared objects only once
if "mp_ctx" not in st.session_state:
    st.session_state.mp_ctx = get_context("spawn")
    st.session_state.lock = st.session_state.mp_ctx.Lock()
    st.session_state.db_lock = st.session_state.mp_ctx.Lock()
    st.session_state.Pid = st.session_state.mp_ctx.Value("i", 0)

mp_ctx = st.session_state.mp_ctx
lock = st.session_state.lock
db_lock = st.session_state.db_lock
Pid = st.session_state.Pid


# Handle start
if start_button:
    inputs = []  # tuples of (type, identifier)

    # convert uploaded files -> temp paths
    if uploaded_files:
        saved_paths = save_uploaded_files(uploaded_files)
        for idx, p in enumerate(saved_paths):
            inputs.append(("file", p))

    # webcams
    for cam in selected_cams:
        inputs.append(("webcam", int(cam)))

    if not inputs:
        st.warning("No inputs selected. Upload videos or choose webcams.")
    else:
        started = 0
        for i, (typ, ident) in enumerate(inputs):
            # unique key for this process
            key = f"proc-{int(time.time())}-{i}-{os.getpid()}"

            if typ == "file":
                vid_path = ident
                cam_id = i
            else:
                vid_path = "webcam"
                cam_id = int(ident)

            proc = mp_ctx.Process(
                target=process_stream,
                args=(vid_path, cam_id, lock, db_lock, Pid),
                daemon=True,
            )
            proc.start()

            st.session_state.processes[key] = proc
            st.session_state.process_meta[key] = {"type": typ, "input": ident, "started_at": time.time()}
            started += 1

        st.success(f"Started {started} process(es). See the Running processes list.")

        emb_worker = mp.Process(target=process_embeddings_job,
                                args=([saved_paths]))
        sync_sql_worker = mp.Process(target=sync_clip_embeddings_to_sql)

        db = SessionLocal()
        lost_persons = get_lost_persons(db)
        if lost_persons:
            check_for_lost_persons = mp.Process(target=lost_person_search, args=(lost_persons,))
            check_for_lost_persons.start()
            st.session_state.processes['check_for_lost_persons'] = check_for_lost_persons
            started += 1

        emb_worker.start()
        sync_sql_worker.start()

        st.session_state.processes['emb_worker'] = emb_worker
        st.session_state.processes['sync_sql_worker'] = sync_sql_worker

        started += 2


# Handle stop
if stop_button:
    n = 0
    for key, proc in list(st.session_state.processes.items()):
        if proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass
            proc.join(timeout=2)
        # cleanup temp files if this was a file
        meta = st.session_state.process_meta.get(key, {})
        if meta.get("type") == "file":
            try:
                os.remove(meta.get("input"))
            except Exception:
                pass
        # remove entries
        st.session_state.processes.pop(key, None)
        st.session_state.process_meta.pop(key, None)
        n += 1

    st.success(f"Stopped {n} process(es)")


# Small heartbeat refresh so the UI updates statuses reasonably frequently
# st.rerun()


st.success("System loaded successfully! Select a section from the sidebar to get started.")
