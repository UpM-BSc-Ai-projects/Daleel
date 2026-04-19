import os
import uuid
import torch
import numpy as np
from io import BytesIO
import streamlit as st
from PIL import Image
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, PointStruct
from sentence_transformers import SentenceTransformer
import cv2
import time
from ultralytics import YOLO
from deep_translator import GoogleTranslator

# Constants
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin123"
BUCKET_NAME = "image-dataset"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "images"
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

# Translations
TRANSLATIONS = {
    "page_title": {"en": "Image Storage & Search", "ar": "تخزين الصور والبحث عنها"},
    "app_title": {"en": "🔍 Multi-Modal Image Search Platform", "ar": "🔍 منصة البحث المتعددة الوسائط للصور"},
    "tab_search": {"en": "🔍 Image Search", "ar": "🔍 البحث عن الصور"},
    "tab_dash": {"en": "📊 Dashboard", "ar": "📊 لوحة التحكم"},
    "tab_capture": {"en": "🎥 Capture & Detect", "ar": "🎥 الالتقاط والكشف"},
    "sidebar_session": {"en": "### 🛠️ Session Management", "ar": "### 🛠️ إدارة الجلسة"},
    "delete_session": {"en": "🗑️ Delete Session Data", "ar": "🗑️ حذف بيانات الجلسة"},
    "delete_success": {"en": "Successfully deleted {} items.", "ar": "تم حذف {} عناصر بنجاح."},
    "delete_error": {"en": "Error deleting session data: {}", "ar": "خطأ في حذف بيانات الجلسة: {}"},
    "no_session": {"en": "No session data to delete.", "ar": "لا توجد بيانات جلسة لحذفها."},
    "search_desc": {"en": "Search across the ingested dataset using **Text Queries**, **Image References**, or recursively use specific images via OpenAI's CLIP model.", 
                   "ar": "ابحث في مجموعة البيانات باستخدام **الاستعلامات النصية**، أو **مراجع الصور**، أو استخدم الصور بشكل تكراري عبر نموذج CLIP."},
    "text_desc_header": {"en": "#### 📝 Text Description", "ar": "#### 📝 وصف نصي"},
    "text_query_label": {"en": "Enter a query to search by text:", "ar": "أدخل استعلاماً للبحث بالنص:"},
    "text_query_placeholder": {"en": "e.g., A tall bald man wearing white thobe...", "ar": "مثلاً: رجل طويل أصلع يرتدي ثوباً أبيض..."},
    "image_ref_header": {"en": "#### 🖼️ Image Reference", "ar": "#### 🖼️ مرجع بصري"},
    "recursive_active": {"en": "🔄 Recursive Search Active", "ar": "🔄 البحث التكراري نشط"},
    "recursive_info": {"en": "Using Image ID `{}` as the query point.", "ar": "يتم استخدام معرف الصورة `{}` كنقطة استعلام."},
    "clear_recursive": {"en": "❌ Clear Recursive Image", "ar": "❌ مسح الصورة التكرارية"},
    "file_uploader_label": {"en": "Alternatively, upload an image to search by visual similarity:", "ar": "بدلاً من ذلك، قم بتحميل صورة للبحث عن طريق التشابه البصري:"},
    "metadata_filters_header": {"en": "#### ⚙️ Metadata Filters", "ar": "#### ⚙️ فلاتر البيانات الوصفية"},
    "metadata_filters_caption": {"en": "Apply logical filtering on database payloads.", "ar": "تطبيق التصفية المنطقية على حمولات قاعدة البيانات."},
    "select_cams": {"en": "Select Camera(s):", "ar": "اختر الكاميرا (الكاميرات):"},
    "select_frames": {"en": "Enter Frame(s) (comma separated):", "ar": "أدخل الإطار (الإطارات) (مفصولة بفواصل):"},
    "score_threshold_header": {"en": "#### 🎯 Score Threshold", "ar": "#### 🎯 عتبة التشابه"},
    "score_threshold_label": {"en": "Minimum Similarity Score:", "ar": "الحد الأدنى لدرجة التشابه:"},
    "execute_search": {"en": "🔍 Execute Search", "ar": "🔍 تنفيذ البحث"},
    "search_warning": {"en": "Please provide either text, an image, or activate a recursive search to continue.", "ar": "يرجى تقديم نص أو صورة أو تفعيل البحث التكراري للمتابعة."},
    "processing_search": {"en": "Processing & vectorizing query...", "ar": "جاري معالجة وتوجيه الاستعلام..."},
    "search_results_header": {"en": "### Search Results", "ar": "### نتائج البحث"},
    "searching_spinner": {"en": "Searching Qdrant Vector DB (Limit: {})...", "ar": "البحث في قاعدة بيانات Qdrant (الحد: {})..."},
    "details_actions": {"en": "Details & Actions", "ar": "التفاصيل والإجراءات"},
    "query_this": {"en": "🔄 Query this Image", "ar": "🔄 البحث بهذه الصورة"},
    "system_diag_header": {"en": "System Diagnostics & Statistics", "ar": "تشخيص وإحصائيات النظام"},
    "refresh_data": {"en": "🔄 Refresh Data", "ar": "🔄 تحديث البيانات"},
    "qdrant_db": {"en": "Qdrant Vector DB", "ar": "قاعدة بيانات Qdrant المتجهة"},
    "minio_storage": {"en": "MinIO Object Storage", "ar": "تخزين الكائنات MinIO"},
    "status": {"en": "Status:", "ar": "الحالة:"},
    "online": {"en": "Online", "ar": "متصل"},
    "offline": {"en": "Offline", "ar": "غير متصل"},
    "total_indexed": {"en": "Total Indexed Vectors", "ar": "إجمالي المتجهات المفهرسة"},
    "est_storage": {"en": "Est. Storage (Raw)", "ar": "التخزين المقدر (خام)"},
    "total_images": {"en": "Total Images Stored", "ar": "إجمالي الصور المخزنة"},
    "total_utilization": {"en": "Total Storage Utilized", "ar": "إجمالي التخزين المستخدم"},
    "realtime_header": {"en": "🎥 Real-time Person Detection & Ingestion", "ar": "🎥 الكشف عن الأشخاص والالتقاط في الوقت الفعلي"},
    "realtime_desc": {"en": "Capture frames every 5 seconds, detect people using YOLO, and automatically index them into the database.", 
                    "ar": "التقاط الإطارات كل 5 ثوانٍ، والكشف عن الأشخاص باستخدام YOLO، وفهرستها تلقائياً في قاعدة البيانات."},
    "video_config_header": {"en": "#### 🛠️ Video Configuration", "ar": "#### 🛠️ تكوين الفيديو"},
    "video_path_label": {"en": "Local Video Path:", "ar": "مسار الفيديو المحلي:"},
    "yolo_conf_label": {"en": "Detection Confidence Threshold:", "ar": "عتبة ثقة الكشف:"},
    "start_processing": {"en": "🚀 Start Processing", "ar": "🚀 بدء المعالجة"},
    "session_summary_header": {"en": "### 📋 Session Summary ({} Crops Ingested)", "ar": "### 📋 ملخص الجلسة (تم التقاط {} صورة)"},
    "loading_model": {"en": "Loading CLIP model...", "ar": "جاري تحميل نموذج CLIP..."},
    "deleting_session": {"en": "Deleting session data...", "ar": "جاري حذف بيانات الجلسة..."},
    "failed_load_id": {"en": "Failed to load image ID:\n{}", "ar": "فشل تحميل معرف الصورة:\n{}"},
    "partial_results": {"en": "Showing partial results.", "ar": "عرض نتائج جزئية."},
    "expand_results": {"en": "🔽 Expand to 100 Results", "ar": "🔽 التوسيع إلى 100 نتيجة"},
    "fetching_db": {"en": "Fetching DB Info...", "ar": "جاري جلب معلومات قاعدة البيانات..."},
    "yolo_not_found": {"en": "YOLO model file `yolo_person_yv11_best.pt` not found in directory.", "ar": "ملف نموذج YOLO لم يتم العثور عليه في المسار."},
    "system_active": {"en": "System active. Monitoring for persons...", "ar": "النظام نشط. مراقبة الأشخاص..."},
    "reached_end": {"en": "Reached end of video file.", "ar": "تم الوصول إلى نهاية ملف الفيديو."},
    "detected_persons": {"en": "[{}s] Detected {} person(s)!", "ar": "[{}ث] تم اكتشاف {} شخصاً!"},
    "scanning": {"en": "[{}s] Scanning... No one found.", "ar": "[{}ث] جاري المسح... لم يتم العثور على أحد."},
    "processing_video_time": {"en": "**Processing Video Time:** {}s", "ar": "**وقت معالجة الفيديو:** {}ث"},
    "latest_captures": {"en": "#### Latest Captures", "ar": "#### أحدث الالتقاطات"},
    "person_num": {"en": "Person {} @ {}s", "ar": "شخص {} @ {}ث"},
}

def t(key):
    lang = st.session_state.get("language", "en")
    return TRANSLATIONS.get(key, {}).get(lang, key)

def apply_rtl():
    if st.session_state.get("language") == "ar":
        st.markdown(
            """
            <style>
            .main {
                direction: rtl;
                text-align: right;
            }
            .stTabs [data-baseweb="tab-list"] {
                direction: rtl;
            }
            .stMultiSelect [data-baseweb="tag"] {
                direction: rtl;
            }
            div[data-testid="stSidebar"] {
                direction: rtl;
            }
            div[data-testid="stExpander"] {
                direction: rtl;
            }
            /* Fix slider alignment for RTL */
            .stSlider [data-testid="stMarkdownContainer"] {
                text-align: right;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(MODEL_NAME, device=device)

@st.cache_resource
def get_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

@st.cache_resource
def load_yolo():
    if os.path.exists("yolo_person_yv11_best.pt"):
        return YOLO("yolo_person_yv11_best.pt")
    return None

@st.cache_data(ttl=600)
def get_available_cameras(_client):
    """Scroll Qdrant to find available unique cameras dynamically."""
    cams = set()
    try:
        offset = None
        for _ in range(5):
            res, offset = _client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=["Cam"],
                with_vectors=False,
                offset=offset
            )
            for p in res:
                if "Cam" in p.payload and p.payload["Cam"] is not None:
                    cams.add(p.payload["Cam"])
            if offset is None:
                break
    except Exception:
        pass
    return sorted(list(cams), key=str)

def fetch_db_stats(minio_c, qdrant_c):
    stats = {
        'qdrant_points': 0,
        'minio_objects': 0,
        'qdrant_status': 'Offline',
        'minio_status': 'Offline',
        'minio_size_bytes': 0,
        'qdrant_est_bytes': 0
    }
    
    try:
        col = qdrant_c.get_collection(COLLECTION_NAME)
        stats['qdrant_points'] = col.points_count
        stats['qdrant_status'] = 'Online'
        # CLIP ViT-B-32 emits 512 dimension floats (4 bytes each)
        stats['qdrant_est_bytes'] = col.points_count * 512 * 4
    except Exception:
        pass

    try:
        objs = list(minio_c.list_objects(BUCKET_NAME))
        stats['minio_objects'] = len(objs)
        stats['minio_size_bytes'] = sum(obj.size for obj in objs)
        stats['minio_status'] = 'Online'
    except Exception:
        pass
            
    return stats


# Streamlit Page Setup
if "language" not in st.session_state:
    st.session_state.language = "en"

st.set_page_config(page_title=t("page_title"), page_icon="🔍", layout="wide")
apply_rtl()

# Header with Language Switch
st.markdown(
    """
    <style>
    .lang-container {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 8px;
        margin-bottom: -10px;
    }
    .lang-text {
        font-weight: bold;
        font-size: 1rem;
        margin: 0;
        padding-top: 5px; /* Adjust for vertical alignment */
    }
    /* Target the toggle specifically to remove its default margin */
    div[data-testid="stWidgetLabel"] {
        display: none;
    }
    .stToggle {
        margin-bottom: -15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with Language Switch
st.markdown(
    """
    <style>
    /* Align everything in the columns to the center vertically */
    [data-testid="column"] {
        display: flex;
        align-items: center;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
    }
    .lang-text {
        font-weight: bold;
        font-size: 1rem;
        margin: 0;
        white-space: nowrap;
    }
    /* Pull language elements to the right and closer */
    .stToggle {
        margin-top: -15px !important;
    }
    div[data-testid="column"]:nth-of-type(2), 
    div[data-testid="column"]:nth-of-type(3), 
    div[data-testid="column"]:nth-of-type(4) {
        justify-content: center !important;
        width: fit-content !important;
        min-width: 0px !important;
        padding: 0 5px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col_title, col_en, col_toggle, col_ar = st.columns([15, 1, 1, 1])

with col_title:
    st.markdown(f'<h1 class="header-title">{t("app_title")}</h1>', unsafe_allow_html=True)
with col_en:
    st.markdown('<p class="lang-text">English</p>', unsafe_allow_html=True)
with col_toggle:
    is_arabic = st.toggle("", value=(st.session_state.language == "ar"), label_visibility="collapsed")
with col_ar:
    st.markdown('<p class="lang-text">العربية</p>', unsafe_allow_html=True)

if is_arabic and st.session_state.language != "ar":
    st.session_state.language = "ar"
    st.rerun()
elif not is_arabic and st.session_state.language != "en":
    st.session_state.language = "en"
    st.rerun()

# State initialization
if "result_limit" not in st.session_state:
    st.session_state.result_limit = 5
if "recursive_query_id" not in st.session_state:
    st.session_state.recursive_query_id = None
if "query_vector" not in st.session_state:
    st.session_state.query_vector = None
if "query_filter" not in st.session_state:
    st.session_state.query_filter = None
if "score_threshold" not in st.session_state:
    st.session_state.score_threshold = 0.0
if "force_search" not in st.session_state:
    st.session_state.force_search = False
if "session_ingested_ids" not in st.session_state:
    st.session_state.session_ingested_ids = []

# Initialize global clients
minio_client = get_minio_client()
qdrant_client = get_qdrant_client()

with st.spinner(t("loading_model")):
    model = load_model()

# Setup Tabs
tab_search, tab_dash, tab_capture = st.tabs([t("tab_search"), t("tab_dash"), t("tab_capture")])

with st.sidebar:
    st.markdown(t("sidebar_session"))
    if st.button(t("delete_session"), type="primary", width="stretch"):
        if st.session_state.session_ingested_ids:
            with st.spinner(t("deleting_session")):
                try:
                    qdrant_client.delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=st.session_state.session_ingested_ids
                    )
                    for image_id in st.session_state.session_ingested_ids:
                        minio_client.remove_object(BUCKET_NAME, f"{image_id}.webp")
                    st.success(t("delete_success").format(len(st.session_state.session_ingested_ids)))
                    st.session_state.session_ingested_ids = []
                except Exception as e:
                    st.error(t("delete_error").format(e))
        else:
            st.info(t("no_session"))


with tab_search:
    st.markdown(t("search_desc"))
    
    st.divider()
    
    # ---------------- 1. QUERY INPUTS ----------------
    col_q, col_f = st.columns([1.5, 1])
    
    with col_q:
        st.markdown(t("text_desc_header"))
        text_query = st.text_input(t("text_query_label"), placeholder=t("text_query_placeholder"))
        
        st.markdown(t("image_ref_header"))
        
        if st.session_state.recursive_query_id:
            st.info(f"**{t('recursive_active')}**\n\n{t('recursive_info').format(st.session_state.recursive_query_id)}")
            if st.button(t("clear_recursive")):
                st.session_state.recursive_query_id = None
                st.session_state.force_search = False
                st.session_state.query_vector = None
                st.rerun()
            uploaded_file = None
        else:
            uploaded_file = st.file_uploader(t("file_uploader_label"), type=["png", "jpg", "jpeg"])

    # ---------------- 2. METADATA FILTERS ----------------
    with col_f:
        st.markdown(t("metadata_filters_header"))
        st.caption(t("metadata_filters_caption"))
        
        avail_cams = get_available_cameras(qdrant_client)
        selected_cams = st.multiselect(t("select_cams"), options=avail_cams)
        
        selected_frames_str = st.text_input(t("select_frames"), placeholder="e.g. 10, 20, 30")
        selected_frames = []
        if selected_frames_str:
            for f in selected_frames_str.split(","):
                f = f.strip()
                if f.isdigit():
                    selected_frames.append(int(f))

    # Evaluate Filters
    filter_conditions = []
    if selected_cams:
        filter_conditions.append(FieldCondition(key="Cam", match=MatchAny(any=selected_cams)))
    if selected_frames:
        filter_conditions.append(FieldCondition(key="Frame", match=MatchAny(any=selected_frames)))

    query_filter = Filter(must=filter_conditions) if filter_conditions else None

    if query_filter:
        st.success(f"Applying filters: {len(filter_conditions)} conditions active.")
        
    st.markdown(t("score_threshold_header"))
    score_threshold = st.slider(t("score_threshold_label"), min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    st.markdown("<br>", unsafe_allow_html=True)

    # Enable Search Action
    if st.button(t("execute_search"), type="primary", width="stretch") or st.session_state.force_search:
        st.session_state.force_search = False
        
        if not text_query and not uploaded_file and not st.session_state.recursive_query_id:
            st.warning(t("search_warning"))
        else:
            # We reset limit on new search explicitly
            st.session_state.result_limit = 5
            st.session_state.query_filter = query_filter
            st.session_state.score_threshold = score_threshold
            
            with st.spinner(t("processing_search")):
                vectors = []
                
                if text_query:
                    # Translate to English for CLIP
                    try:
                        translated_query = GoogleTranslator(source='auto', target='en').translate(text_query)
                        if translated_query != text_query:
                            st.caption(f"Searching for: {translated_query}")
                        text_vector = model.encode(translated_query)
                    except Exception:
                        text_vector = model.encode(text_query)
                    vectors.append(text_vector)
                
                if st.session_state.recursive_query_id:
                    # Fetch recursive vector
                    res = qdrant_client.retrieve(collection_name=COLLECTION_NAME, ids=[st.session_state.recursive_query_id], with_vectors=True)
                    if res and res[0].vector:
                        vectors.append(res[0].vector)
                elif uploaded_file is not None:
                    img = Image.open(uploaded_file)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    st.image(img, caption="Query Image", width=150)
                    vectors.append(model.encode(img))
                    
                if len(vectors) > 1:
                    combined = np.mean(vectors, axis=0)
                    st.session_state.query_vector = combined.tolist()
                elif len(vectors) == 1:
                    st.session_state.query_vector = vectors[0] if isinstance(vectors[0], list) else vectors[0].tolist()
                else:
                    st.session_state.query_vector = None

    # ---------------- 3. EXECUTE SEARCH ----------------
    if st.session_state.query_vector:
        st.markdown(t("search_results_header"))
        with st.spinner(t("searching_spinner").format(st.session_state.result_limit)):
            try:
                # Prepare threshold optimally
                thresh = st.session_state.score_threshold if st.session_state.score_threshold > 0.0 else None
                
                search_response = qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=st.session_state.query_vector,
                    query_filter=st.session_state.query_filter,
                    limit=st.session_state.result_limit,
                    score_threshold=thresh
                )
                search_results = search_response.points
                
                if search_results:
                    # Provide a scrollable container for the results to prevent crowd stretching
                    with st.container(height=800, border=True):
                        # Distribute elegantly into chunks (5 per row uniformly)
                        for chunk_idx in range(0, len(search_results), 5):
                            chunk = search_results[chunk_idx:chunk_idx+5]
                            cols = st.columns(5)
                            
                            for i, result in enumerate(chunk):
                                image_id = result.id
                                score = result.score
                                orig_filename = result.payload.get("filename", "Unknown")
                                cam = result.payload.get("Cam", "N/A")
                                frame = result.payload.get("Frame", "N/A")
                                
                                try:
                                    response = minio_client.get_object(BUCKET_NAME, f"{image_id}.webp")
                                    img_data = BytesIO(response.read())
                                    response.close()
                                    response.release_conn()
                                    result_img = Image.open(img_data)
                                    
                                    with cols[i]:
                                        st.image(result_img, caption=f"Score: {score:.3f} | C:{cam} F:{frame}", width="stretch")
                                        with st.expander(t("details_actions")):
                                            st.caption(f"**ID:** `{image_id[:12]}...`")
                                            st.caption(f"**File:** {orig_filename}")
                                            
                                            if st.button(t("query_this"), key=f"query_{image_id}_{chunk_idx}_{i}"):
                                                st.session_state.recursive_query_id = image_id
                                                st.session_state.force_search = True
                                                st.rerun()
                                                
                                except Exception:
                                    with cols[i]:
                                        st.error(t("failed_load_id").format(image_id))
                    
                    # Show Expansion Button only if we hit the limit, and the limit isn't already 100
                    st.divider()
                    if len(search_results) >= st.session_state.result_limit and st.session_state.result_limit < 100:
                        st.markdown(f"<p style='text-align: center;'>{t('partial_results')}</p>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns([1,1,1])
                        with col2:
                            if st.button(t("expand_results"), width="stretch"):
                                st.session_state.result_limit = 100
                                st.rerun()

                else:
                    st.info("No matching results found. Adjust filters or query.")
            except Exception as e:
                st.error(f"Error during Qdrant Search: {e}")


with tab_dash:
    st.header(t("system_diag_header"))
    
    col_btn, empty = st.columns([1, 4])
    with col_btn:
        st.button(t("refresh_data"), type="secondary", width="stretch")

    with st.spinner(t("fetching_db")):
        stats = fetch_db_stats(minio_client, qdrant_client)
        
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t("qdrant_db"))
        qd_status = stats['qdrant_status']
        st.markdown(f"**{t('status')}** {'🟢 ' + t('online') if qd_status == 'Online' else '🔴 ' + t('offline')}")
        
        c1, c2 = st.columns(2)
        c1.metric(label=t("total_indexed"), value=f"{stats['qdrant_points']:,}")
        est_mb = stats['qdrant_est_bytes'] / (1024 * 1024)
        c2.metric(label=t("est_storage"), value=f"{est_mb:.2f} MB")
        
        per_vector_bytes = (stats['qdrant_est_bytes'] / stats['qdrant_points']) if stats['qdrant_points'] else 0
        st.caption(f"⚖️ ~{per_vector_bytes:.0f} bytes per raw vector")
        
    with col2:
        st.subheader(t("minio_storage"))
        min_status = stats['minio_status']
        st.markdown(f"**{t('status')}** {'🟢 ' + t('online') if min_status == 'Online' else '🔴 ' + t('offline')}")
        
        c1, c2 = st.columns(2)
        c1.metric(label=t("total_images"), value=f"{stats['minio_objects']:,}")
        minio_mb = stats['minio_size_bytes'] / (1024 * 1024)
        c2.metric(label=t("total_utilization"), value=f"{minio_mb:.2f} MB")
        
        per_image_kb = (stats['minio_size_bytes'] / stats['minio_objects'] / 1024) if stats['minio_objects'] else 0
        st.caption(f"⚖️ ~{per_image_kb:.1f} KB average size per encoded image")

with tab_capture:
    st.header(t("realtime_header"))
    st.markdown(t("realtime_desc"))
    
    yolo_model = load_yolo()
    if not yolo_model:
        st.error(t("yolo_not_found"))
        st.stop()
        
    st.markdown(t("video_config_header"))
    video_value = r"C:\Users\themi\PycharmProjects\Capstone2\Datasets\UPM DL Lab.mp4"
    video_path = st.text_input(t("video_path_label"), value=video_value)
    yolo_conf = st.slider(t("yolo_conf_label"), min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    run_capture = st.toggle(t("start_processing"), key="run_capture")
    
    col_status, col_preview = st.columns([1, 2])
    
    status_area = col_status.empty()
    preview_area = col_preview.empty()
    
    if run_capture:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error(f"Could not open video source: {video_path}")
        else:
            last_processed_msec = -5000 
            
            status_area.info("System active. Monitoring for persons...")
            
            while st.session_state.run_capture:
                target_msec = last_processed_msec + 5000
                cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
                
                ret, frame = cap.read()
                if not ret:
                    status_area.warning(t("reached_end"))
                    break
                
                current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                current_sec = int(current_msec / 1000)
                last_processed_msec = current_msec
                
                status_area.markdown(t("processing_video_time").format(current_sec))
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                
                # Run YOLO
                results = yolo_model(frame, classes=[0], verbose=False, conf=yolo_conf) 
                
                found_people = False
                for result in results:
                    boxes = result.boxes
                    if len(boxes) > 0:
                        found_people = True
                        status_area.success(t("detected_persons").format(current_sec, len(boxes)))
                        
                        with preview_area.container():
                            st.markdown(t("latest_captures"))
                            cols = st.columns(min(len(boxes), 3))
                            
                            for i, box in enumerate(boxes):
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                crop = pil_frame.crop((x1, y1, x2, y2))
                                image_id = str(uuid.uuid4())
                                vector = model.encode(crop).tolist()
                                
                                img_byte_arr = BytesIO()
                                crop.save(img_byte_arr, format='WEBP', quality=60)
                                img_byte_arr.seek(0)
                                img_size = img_byte_arr.getbuffer().nbytes
                                
                                minio_client.put_object(
                                    BUCKET_NAME,
                                    f"{image_id}.webp",
                                    img_byte_arr,
                                    length=img_size,
                                    content_type="image/webp"
                                )
                                
                                qdrant_client.upsert(
                                    collection_name=COLLECTION_NAME,
                                    points=[PointStruct(
                                        id=image_id,
                                        vector=vector,
                                        payload={"Cam": "Live", "Frame": f"Time_{current_sec}s", "session": "active"}
                                    )]
                                )
                                
                                st.session_state.session_ingested_ids.append(image_id)
                                
                                if i < 3: 
                                    cols[i].image(crop, caption=t("person_num").format(i+1, current_sec), width="stretch")
                
                if not found_people:
                    status_area.info(t("scanning").format(current_sec))
                    preview_area.empty()
                
                time.sleep(0.1)
                
            cap.release()

    # --- SESSION SUMMARY GALLERY ---
    if not run_capture and st.session_state.session_ingested_ids:
        st.divider()
        st.markdown(t("session_summary_header").format(len(st.session_state.session_ingested_ids)))
        
        # Display in a grid
        cols_per_row = 4
        ids = st.session_state.session_ingested_ids[::-1] # Show newest first
        
        for i in range(0, len(ids), cols_per_row):
            batch = ids[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            
            for j, image_id in enumerate(batch):
                try:
                    # Fetch from MinIO
                    response = minio_client.get_object(BUCKET_NAME, f"{image_id}.webp")
                    img_data = BytesIO(response.read())
                    response.close()
                    response.release_conn()
                    img = Image.open(img_data)
                    
                    with cols[j]:
                        st.image(img, width="stretch")
                        st.caption(f"**ID:** `{image_id[:8]}...`")
                        # Add a delete individual button if helpful, but for now just info
                except Exception:
                    with cols[j]:
                        st.error(f"Error loading {image_id[:8]}")

