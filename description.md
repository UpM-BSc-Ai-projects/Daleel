# AI-Powered Lost Persons Management System - Complete Technical Architecture

## Table of Contents

- [AI-Powered Lost Persons Management System - Complete Technical Architecture](#ai-powered-lost-persons-management-system---complete-technical-architecture)
- [System Overview](#system-overview)

- [PART 1: System Initialization Sequence](#part-1-system-initialization-sequence)
  - [1.1 Application Entry Point: Streamlit App Initialization](#11-application-entry-point-streamlit-app-initialization)
  - [1.2 Database Schema Architecture](#12-database-schema-architecture)
  - [1.3 Vector Database Collections](#13-vector-database-collections)

- [PART 2: Video Stream Processing Pipeline](#part-2-video-stream-processing-pipeline)
  - [2.1 Stream Processing Initiation](#21-stream-processing-initiation)
  - [2.2 Deep Stream Processing: MCDPT Module](#22-deep-stream-processing-mcdpt-module)

- [PART 3: Embedding Processing Pipeline](#part-3-embedding-processing-pipeline)
  - [3.1 Embedding Worker Process](#31-embedding-worker-process)
  - [3.2 SQL Synchronization Worker](#32-sql-synchronization-worker)

- [PART 4: Streamlit Frontend Pages](#part-4-streamlit-frontend-pages)
  - [4.1 Page Structure](#41-page-structure)
  - [4.2 CRUD Operations](#42-crud-operations)

- [PART 5: CLIP Module – Text-Based Person Retrieval](#part-5-clip-module---text-based-person-retrieval)
  - [5.1 CLIP Class](#51-clip-class)

- [PART 6: Data Flow and System States](#part-6-data-flow-and-system-states)
  - [6.1 Complete Data Flow Diagram](#61-complete-data-flow-diagram)
  - [6.2 System States and Transitions](#62-system-states-and-transitions)
  - [6.3 Concurrency and Synchronization](#63-concurrency-and-synchronization)

- [PART 7: Key Algorithms and Models](#part-7-key-algorithms-and-models)
  - [7.1 Person Detection: YOLO RT-DETR-L](#71-person-detection-yolo-rt-detr-l)
  - [7.2 Person Re-Identification: OSNet](#72-person-re-identification-osnet)
  - [7.3 Multi-Camera Tracking: DeepSort+](#73-multi-camera-tracking-deepsort)
  - [7.4 Emotion Detection: ViT-Face-Expression](#74-emotion-detection-vit-face-expression)
  - [7.5 Vision-Language Embeddings: CLIP](#75-vision-language-embeddings-clip)
  - [7.6 Attribute Detection: Similarity-based Classification](#76-attribute-detection-similarity-based-classification)

- [PART 8: Database Operations Flow](#part-8-database-operations-flow)
  - [8.1 Person Registration Flow](#81-person-registration-flow)
  - [8.2 Detection-to-Database Flow](#82-detection-to-database-flow)
  - [8.3 Search Flow](#83-search-flow)

- [PART 9: Configuration and Parameters](#part-9-configuration-and-parameters)
  - [9.1 Detection Parameters](#91-detection-parameters)
  - [9.2 Embedding Parameters](#92-embedding-parameters)
  - [9.3 Processing Parameters](#93-processing-parameters)

- [PART 10: Error Handling and Edge Cases](#part-10-error-handling-and-edge-cases)
  - [10.1 Video Processing Edge Cases](#101-video-processing-edge-cases)
  - [10.2 Database Edge Cases](#102-database-edge-cases)
  - [10.3 Embedding Edge Cases](#103-embedding-edge-cases)

- [PART 11: Performance Considerations](#part-11-performance-considerations)
  - [11.1 GPU Utilization](#111-gpu-utilization)
  - [11.2 Memory Management](#112-memory-management)
  - [11.3 Scalability Limitations](#113-scalability-limitations)

- [PART 12: Deployment and Setup](#part-12-deployment-and-setup)
  - [12.1 Required Services](#121-required-services)
  - [12.2 File Structure](#122-file-structure)
  - [12.3 Startup Command](#123-startup-command)

- [PART 13: Summary of System Flow](#part-13-summary-of-system-flow)

- [PART 14: Key Technical Insights](#part-14-key-technical-insights)
  - [14.1 Why This Architecture?](#141-why-this-architecture)
  - [14.2 Bottlenecks](#142-bottlenecks)
  - [14.3 Future Improvements](#143-future-improvements)

- [Conclusion](#conclusion)


## System Overview

This is a comprehensive AI-powered security support system designed for the Prophet's Mosque (Al-Masjid An-Nabawi) to identify, locate, and manage lost or disoriented individuals in real-time using advanced computer vision, deep learning, and person re-identification technologies.

---

## PART 1: SYSTEM INITIALIZATION SEQUENCE

### 1.1 Application Entry Point: Streamlit App Initialization

**File:** `/Streamlit/app.py` (Lines 1-51)

When the Streamlit application starts, the following initialization sequence occurs:

#### Step 1: Page Configuration
- Streamlit sets the page title to "Lost Persons Management System"
- Page layout is set to "wide" for better UI space utilization
- Page icon is configured

#### Step 2: Session State Initialization (Lines 24-41)
The `ensure_session_state()` function initializes all session-level variables that persist across Streamlit reruns:

- **processes dict**: Maps process IDs to multiprocessing.Process objects for tracking active video processing threads
- **process_meta dict**: Stores metadata about each process (type: 'file' or 'webcam', input identifier, start time)
- **mp_ctx**: Multiprocessing context using "spawn" method (creates fresh Python interpreter for each process)
- **lock**: Thread synchronization lock for GPU access (prevents concurrent CUDA operations)
- **db_lock**: Database synchronization lock for preventing concurrent database writes
- **Pid**: Shared integer value tracking the next available point ID for Qdrant vector database insertions
- **db_initialized**: Boolean flag ensuring database tables are created only once
- **collections_initialized**: Boolean flag ensuring Qdrant vector collections are created only once

#### Step 3: Database Initialization (Lines 44-46)
- Calls `create_tables()` from `models.py`
- This function uses SQLAlchemy ORM to create all database tables in SQLite if they don't exist
- Tables created: admins, security_staff, family_members, persons, camera_detected_persons, last_seen, search_data, results_list

#### Step 4: Vector Database Initialization (Lines 49-51)
- Calls `initialize_collections()` from `models.py`
- Connects to Qdrant vector database at `http://localhost:6333`
- Creates two collections if they don't exist:
  - **CLIP_embeddings**: Stores 512-dimensional CLIP embeddings of detected persons with metadata (elderly, disabled, lost status)
  - **stream**: Stores real-time tracking embeddings from video stream processing with coordinates and camera IDs

---

### 1.2 Database Schema Architecture

**File:** `/Streamlit/models.py`

The system uses 8 interconnected SQLAlchemy ORM models:

#### Admin Table
- **Fields**: adminId (PK), name, email, passwordHash, phoneNumber
- **Purpose**: System administrators who manage security staff
- **Relationships**: One-to-many with SecurityStaff (backref: security_staffs)

#### SecurityStaff Table
- **Fields**: staffId (PK), name, email, passwordHash, phoneNumber, admin_id (FK)
- **Purpose**: Security personnel who report missing persons and initiate searches
- **Relationships**: 
  - Many-to-one with Admin (backref: security_staffs)
  - One-to-many with Person (backref: reported_persons)
  - One-to-many with SearchData (backref: search_requests)

#### FamilyMember Table
- **Fields**: familyMemberId (PK), firstName, lastName, idNumber, phoneNumber, relation
- **Purpose**: Family members reporting missing persons
- **Relationships**: One-to-many with Person (backref: persons)

#### CameraDetectedPerson Table
- **Fields**: cameraDetectedPersonId (PK, String), potentiallyLost (Boolean), isElderly (Boolean), isDisabled (Boolean)
- **Purpose**: Persons automatically detected by AI in camera feeds
- **Relationships**: One-to-many with Person, LastSeen, ResultsList

#### Person Table
- **Fields**: personId (PK), firstName, lastName, age, isMale, idNumber, phoneNumber, LastLocation, description, video, image_1-5, uploadTime, isLost, cameraDetectedPersonId (FK), relatTo_id (FK), reportedBy_id (FK)
- **Purpose**: Core entity representing missing persons with full profile and media
- **Relationships**:
  - Many-to-one with CameraDetectedPerson (backref: persons)
  - Many-to-one with FamilyMember (backref: persons) - CASCADE delete
  - Many-to-one with SecurityStaff (backref: reported_persons)

#### LastSeen Table
- **Fields**: id (PK), location, time, coordinates (JSON), CDPid (FK)
- **Purpose**: Tracks last known location and timestamp of detected persons
- **Relationships**: Many-to-one with CameraDetectedPerson (backref: last_seen_records) - CASCADE delete

#### SearchData Table
- **Fields**: searchID (PK), requestTime, isProcessing (Boolean), isFound (Boolean), requestedBy_id (FK)
- **Purpose**: Tracks search requests initiated by security staff
- **Relationships**: Many-to-one with SecurityStaff (backref: search_requests)

#### ResultsList Table
- **Fields**: id (PK), isAccepted (Boolean), searchID (FK), cameraDetectedPersonId (FK)
- **Purpose**: Stores search results linking detected persons to search requests
- **Relationships**: 
  - Many-to-one with SearchData (backref: results) - CASCADE delete
  - Many-to-one with CameraDetectedPerson (backref: search_results) - CASCADE delete

---

### 1.3 Vector Database Collections

**Qdrant Vector Database** (`http://localhost:6333`)

#### CLIP_embeddings Collection
- **Vector Size**: 512 dimensions
- **Distance Metric**: COSINE similarity
- **Point Structure**:
  ```
  {
    id: unique_integer,
    vector: [512-dim embedding],
    payload: {
      "Pid": global_person_id,
      "is_lost": boolean,
      "is_elderly": boolean,
      "is_disabled": boolean
    }
  }
  ```
- **Purpose**: Stores CLIP vision-language embeddings for person re-identification and attribute classification

#### stream Collection
- **Vector Size**: 512 dimensions
- **Distance Metric**: COSINE similarity
- **Point Structure**:
  ```
  {
    id: unique_integer,
    vector: [512-dim OSNet embedding],
    payload: {
      "Pid": global_track_id,
      "coords": [x1, y1, x2, y2],
      "cam_id": camera_id,
      "frame_count": frame_number,
      "cross_cam": boolean
    }
  }
  ```
- **Purpose**: Stores real-time person embeddings from video streams for tracking across cameras

---

## PART 2: VIDEO STREAM PROCESSING PIPELINE

### 2.1 Stream Processing Initiation

**File:** `/Streamlit/app.py` (Lines 103-192)

#### User Interface for Stream Selection
The main app page displays:
1. **File Uploader**: Accepts MP4, MOV, AVI video files (multiple files supported)
2. **Webcam Selector**: Multiselect dropdown for webcam IDs (0-5)
3. **Start Button**: Initiates processing of selected inputs
4. **Stop Button**: Terminates all active processes
5. **Running Processes Panel**: Real-time display of active processes with status

#### Process Startup Logic (Lines 141-192)

When user clicks "Start Selected":

1. **Input Collection** (Lines 142-152):
   - Uploaded files are saved to temporary files using `save_uploaded_files()`
   - Each file gets a unique temporary path
   - Webcam selections are converted to integer IDs
   - All inputs are collected as tuples: (type, identifier)

2. **Process Creation** (Lines 157-178):
   - For each input, a unique process key is generated: `proc-{timestamp}-{index}-{pid}`
   - A new multiprocessing Process is created with target function `process_stream` from MCDPT module
   - Arguments passed to process_stream:
     - `vid_path`: Path to video file or "webcam" string
     - `cam_id`: Camera/input identifier
     - `lock`: GPU synchronization lock
     - `db_lock`: Database synchronization lock
     - `Pid`: Shared integer for vector ID allocation
   - Process is set as daemon (killed when main process exits)
   - Process is started immediately
   - Metadata is stored in session state

3. **Embedding Worker Processes** (Lines 182-191):
   - Two additional worker processes are spawned:
     - **emb_worker**: Processes embeddings from video frames
     - **sync_sql_worker**: Synchronizes Qdrant embeddings to SQL database
   - Both workers are added to the processes tracking dictionary

---

### 2.2 Deep Stream Processing: MCDPT Module

**File:** `/MCDPT/multiprocessing_mct.py` (Lines 85-186)

This is the core video processing engine that runs in a separate process for each video/webcam input.

#### Initialization (Lines 85-102)

```python
def process_stream(vid_path: str, cam_id: int, lock, db_lock=None, Pid=None)
```

1. **Qdrant Client Connection**:
   - Connects to Qdrant at `http://localhost:6333`
   - Will upsert points to "stream" collection

2. **Re-ID Model Loading**:
   - Loads OSNet (OSNet_x0_25_msmt17) pre-trained weights
   - OSNet is a lightweight person re-identification model
   - Wrapped in custom `OSNet` class that:
     - Loads model to CUDA GPU
     - Applies data transforms: resize to 256x128, normalize with ImageNet stats
     - Returns normalized embeddings (L2 norm = 1)

3. **Object Detector Initialization**:
   - Loads YOLO RT-DETR-L model (real-time detection transformer)
   - Model file: `rtdetr-l.pt`
   - Moved to GPU and set to eval mode
   - Detects 80 COCO classes, filters for class 0 (person)

4. **Tracker Initialization**:
   - Creates `DeepSortPlus` tracker instance
   - Configuration parameters:
     - `global_match_thresh=0.4`: Similarity threshold for matching detections
     - `sct_config`: Multi-camera tracking configuration
   - Tracker maintains persistent IDs across frames

#### Main Processing Loop (Lines 103-182)

The loop processes video frame-by-frame:

```
while True:
  1. Read frame from video
  2. Every 2nd frame: Run object detection
  3. Filter detections to persons only (class 0)
  4. Filter by confidence score >= 0.5
  5. Get bounding boxes [x1, y1, x2, y2]
  6. Run tracker.process() with frame and boxes
  7. For each tracked object:
     a. Extract features
     b. Get global track ID
     c. Draw bounding box with color based on ID
     d. Extract person crop from frame
     e. Generate CLIP embedding
     f. Upsert to Qdrant with metadata
  8. Display frame with annotations
  9. Break on 'q' key press
```

**Frame-by-Frame Processing Details:**

1. **Detection Phase** (Lines 113-126):
   - Runs every 2 frames (frame skip for performance)
   - Uses GPU lock to synchronize CUDA operations
   - YOLO predicts on frame
   - Extracts boxes, classes, confidence scores
   - Filters: only persons (class == 0) with confidence >= 0.5
   - Result: numpy array of bounding boxes [x1, y1, x2, y2]

2. **Tracking Phase** (Lines 129-132):
   - Calls `tracker.process(frame, [boxes])`
   - Tracker uses DeepSort algorithm with OSNet embeddings
   - Maintains consistent IDs across frames
   - Handles occlusions and re-entries

3. **Feature Extraction & Storage** (Lines 133-169):
   - Gets tracked objects with features from tracker
   - For each visible object with features:
     - Extracts bounding box coordinates
     - Gets global track ID (format: "{cam_id}-{local_id}")
     - Generates color based on ID for visualization
     - Draws rectangle and ID text on frame
     - **Critical Section (db_lock)**:
       - Extracts person crop from frame using coordinates
       - Generates OSNet embeddings for each feature
       - Creates PointStruct objects with:
         - Unique ID from shared Pid counter
         - OSNet embedding vector
         - Payload with Pid, coordinates, camera ID, frame count, cross-camera flag
       - Upserts to Qdrant "stream" collection
       - Increments shared Pid counter

4. **Visualization** (Lines 178-180):
   - Displays annotated frame with bounding boxes and IDs
   - Press 'q' to exit

#### Cleanup (Lines 183-185)
- Releases video capture
- Closes all OpenCV windows

---

## PART 3: EMBEDDING PROCESSING PIPELINE

### 3.1 Embedding Worker Process

**File:** `/Embedding_worker/embedding_worker.py`

This process runs in parallel with stream processing to enrich embeddings with AI attributes.

#### Initialization (Lines 163-176)

```python
def process_embeddings_job(paths)
```

1. **Model Loading**:
   - **CLIP Model**: ViT-B/32 variant loaded on GPU
     - 512-dimensional vision-language embeddings
     - Can encode both images and text descriptions
   
   - **Emotion Classifier**: `trpakov/vit-face-expression` from HuggingFace
     - Classifies facial expressions: happy, sad, fear, angry, surprise, neutral
     - Used to detect lost/disoriented persons
   
   - **Text Embeddings for Attributes**:
     - Encodes two text descriptions using CLIP:
       - "an elderly person over 60 years old"
       - "a person in a wheel chair"
     - These embeddings serve as reference vectors for similarity comparison

2. **Video Mapping**:
   - Creates dictionary mapping camera IDs to video file paths
   - Used to extract frames when processing Qdrant records

#### Processing Loop (Lines 178-187)

```
while True:
  1. Sleep 5 seconds (allows stream processing to accumulate records)
  2. Retrieve up to 300 new records from "stream" collection
  3. For each record:
     a. Extract metadata (cam_id, coords, frame_count, Pid)
     b. Extract frame from video at frame_count
     c. Crop person from frame using coordinates
     d. Run emotion classifier on crop
     e. Determine if lost: emotion in [sad, fear, angry] AND confidence > 0.6
     f. Generate CLIP embedding from crop
     g. Compare with elderly/disabled reference embeddings
     h. Upsert to CLIP_embeddings collection with attributes
  4. Increment limit by 300
```

**Detailed Attribute Detection:**

1. **Emotion-based Lost Detection** (Lines 144-145):
   ```python
   emotion_result = emotion_clf(crop_image, top_k=1)[0]
   is_lost = emotion_result["label"].lower() in ["sad", "fear", "angry"] 
             and emotion_result["score"] > 0.6
   ```
   - Runs vision transformer on person crop
   - Gets top emotion prediction
   - Marks as "potentially lost" if negative emotion with high confidence

2. **Elderly Detection** (Lines 96-101, 149):
   ```python
   def determine_elderly(elderly_embedding, person_emb, threshold=0.19):
     cos_sim = np.dot(person_emb, elderly_embedding)
     return cos_sim > threshold
   ```
   - Computes cosine similarity between person CLIP embedding and elderly reference
   - Threshold: 0.19 (tuned for high precision)

3. **Disability Detection** (Lines 105-111, 150):
   ```python
   def determine_disabled(disabled_embedding, person_emb, threshold=0.19):
     cos_sim = np.dot(person_emb, disabled_embedding)
     return cos_sim > threshold
   ```
   - Computes cosine similarity with wheelchair reference
   - Same threshold for consistency

#### Qdrant Upsert (Lines 154-160)
```python
client.upsert(collection_name=CLIP_COLLECTION,
              points=[PointStruct(id=rec.id,
                                vector=clip_emb,
                                payload={"Pid": pid,
                                         "is_lost": is_lost,
                                         "is_elderly": is_elderly,
                                         "is_disabled": is_disabled})])
```
- Updates "CLIP_embeddings" collection with enriched metadata

---

### 3.2 SQL Synchronization Worker

**File:** `/Embedding_worker/embedding_worker.py` (Lines 24-94)

```python
def sync_clip_embeddings_to_sql(interval_seconds: int = 30)
```

This process continuously syncs Qdrant data to SQL database.

#### Synchronization Loop (Lines 33-94)

```
while True:
  1. Query CLIP_embeddings collection for records (IDs: num_synced to num_synced+1000)
  2. For each record:
     a. Extract payload: Pid, is_lost, is_elderly, is_disabled
     b. Check if CameraDetectedPerson exists in SQL with this Pid
     c. If exists: Update attributes
     d. If not exists: Create new CameraDetectedPerson record
  3. Increment num_synced counter
  4. Sleep 30 seconds
```

**Database Operations:**

1. **Check Existing** (Lines 57-59):
   ```python
   existing = db.query(CameraDetectedPerson).filter(
     CameraDetectedPerson.cameraDetectedPersonId == pid
   ).first()
   ```

2. **Update** (Lines 63-69):
   - Calls `update_camera_detected_person()` from CRUD module
   - Updates potentiallyLost, isElderly, isDisabled flags

3. **Create** (Lines 74-82):
   - Creates new CameraDetectedPerson record
   - Uses Qdrant ID as primary key
   - Commits to database

---

## PART 4: STREAMLIT FRONTEND PAGES

### 4.1 Page Structure

**Directory:** `/Streamlit/pages/`

The application has 8 management pages accessible from sidebar navigation:

#### 1_Admin.py - Administrator Management
- Create, read, update, delete admin accounts
- Manage admin-staff relationships
- View admin statistics

#### 2_SecurityStaff.py - Security Personnel Management
- Create, read, update, delete security staff
- Assign staff to admins
- Track staff-reported persons
- View staff statistics

#### 3_FamilyMember.py - Family Member Management
- Register family members reporting missing persons
- Store family contact information and relation type
- Link families to missing persons

#### 4_Person.py - Missing Persons Management (Most Complex)
**File:** `/Streamlit/pages/4_Person.py`

This is the central hub for managing missing persons with comprehensive features:

**Tab 1: Persons List**
- Displays all persons in a DataFrame with columns:
  - ID, Name, Age, Gender, ID Number, Phone, Location, Status
- Shows person details with media:
  - Personal information (name, age, gender, ID, phone, location)
  - Status (Lost/Found)
  - Upload timestamp
  - Family member information (if reported by family)
  - Security staff information (if reported by staff)
  - AI attributes from CameraDetectedPerson
  - Up to 5 images stored in database

**Tab 2: Add Person**
- Form to register new missing person
- Fields:
  - First name, last name, age, gender, ID number, phone
  - Last known location, description
  - Video upload (MP4, MOV, AVI)
  - Image uploads (up to 5 images)
  - Reporter selection (family member or security staff)
- Validation using `validate_person_creation()`:
  - Checks for duplicate phone numbers
  - Checks for duplicate ID numbers
  - Checks for duplicate name+gender combinations
  - Provides detailed error messages with existing person info

**Tab 3: Quick Search**
- Search persons by:
  - Name (first/last)
  - Phone number
  - ID number
  - Location
- Returns matching persons with details

**Tab 4: Update Person**
- Select person from dropdown
- Modify any field
- Update status (Lost/Found)
- Update location
- Add/remove images

**Tab 5: Special Actions**
- Mark person as found
- Update last seen location
- Link to camera detected person
- Bulk operations

**Tab 6: Statistics**
- Total persons count
- Lost vs Found breakdown
- Persons by age group
- Persons by gender
- Recent registrations

#### 5_CameraDetectedPerson.py - AI-Detected Persons
- View persons automatically detected in camera feeds
- Filter by attributes:
  - Potentially lost (emotion-based)
  - Elderly (age-based)
  - Disabled (wheelchair detection)
- View detection metadata:
  - Camera ID
  - Frame count
  - Coordinates
  - Embeddings
- Link detected persons to missing persons

#### 6_LastSeen.py - Location Tracking
- View last known locations of detected persons
- Display location, timestamp, coordinates
- Map visualization (if coordinates available)
- Filter by camera
- Filter by time range

#### 7_SearchData.py - Search Request Management
- Create search requests for missing persons
- Track search status (processing/completed)
- Track if person found
- View search results
- Assign searches to security staff

#### 8_ResultsList.py - Search Results
- View all search results
- Link detected persons to search requests
- Accept/reject matches
- View match confidence scores
- Track which results led to finding persons

---

### 4.2 CRUD Operations

**File:** `/Streamlit/crud.py`

Comprehensive database operations for all entities:

#### Admin CRUD (Lines 10-50)
- `create_admin()`: Create new admin with credentials
- `get_admins()`: Paginated retrieval
- `get_admin_by_id()`: Single admin lookup
- `get_admin_with_staff()`: Admin with related staff
- `update_admin()`: Modify admin fields
- `delete_admin()`: Remove admin
- `get_admins_count()`: Count total
- `get_recent_admins()`: Get latest N admins

#### SecurityStaff CRUD (Lines 55-95)
- `create_security_staff()`: Create staff with admin assignment
- `get_security_staffs()`: Paginated retrieval
- `get_security_staff_by_id()`: Single lookup
- `get_security_staff_with_relations()`: Staff with reported persons and searches
- `update_security_staff()`: Modify staff
- `delete_security_staff()`: Remove staff
- `get_security_staff_count()`: Count total
- `get_recent_security_staff()`: Get latest N staff

#### FamilyMember CRUD (Lines 100+)
- `create_family_member()`: Register family member
- `get_family_members()`: Paginated retrieval
- `get_family_member_by_id()`: Single lookup
- `update_family_member()`: Modify family member
- `delete_family_member()`: Remove family member
- `get_family_members_count()`: Count total
- `get_recent_family_members()`: Get latest N members

#### Person CRUD (Complex)
- `create_person()`: Create missing person with full profile
  - Validates using `validate_person_creation()`
  - Handles image uploads
  - Handles video uploads
  - Links to family/staff reporter
  - Links to camera detected person if available
  
- `get_persons()`: Paginated retrieval
- `get_person_by_id()`: Single lookup
- `get_person_with_relations()`: Person with all related entities
- `update_person()`: Modify person fields
- `delete_person()`: Remove person (cascades to search results)
- `get_persons_count()`: Count total
- `get_recent_persons()`: Get latest N persons
- `get_lost_persons()`: Filter by lost status
- `search_persons()`: Search by name, phone, ID, location

#### CameraDetectedPerson CRUD
- `create_camera_detected_person()`: Create from Qdrant sync
- `get_camera_detected_persons()`: Paginated retrieval
- `get_camera_detected_person_by_id()`: Single lookup
- `update_camera_detected_person()`: Update attributes (elderly, disabled, lost)
- `delete_camera_detected_person()`: Remove (cascades to LastSeen, ResultsList)
- `get_camera_detected_persons_count()`: Count total
- `get_elderly_persons()`: Filter elderly
- `get_disabled_persons()`: Filter disabled
- `get_potentially_lost_persons()`: Filter by emotion

#### LastSeen CRUD
- `create_last_seen()`: Record detection location
- `get_last_seen()`: Paginated retrieval
- `get_last_seen_by_id()`: Single lookup
- `get_last_seen_by_cdp_id()`: Get locations for specific person
- `update_last_seen()`: Update location/time
- `delete_last_seen()`: Remove record
- `get_last_seen_count()`: Count total
- `get_recent_last_seen()`: Get latest N records

#### SearchData CRUD
- `create_search_data()`: Create search request
- `get_search_data()`: Paginated retrieval
- `get_search_data_by_id()`: Single lookup
- `update_search_data()`: Update status/found flag
- `delete_search_data()`: Remove search
- `get_search_data_count()`: Count total
- `get_active_searches()`: Filter by isProcessing=True
- `get_completed_searches()`: Filter by isProcessing=False

#### ResultsList CRUD
- `create_results_list()`: Create search result
- `get_results_list()`: Paginated retrieval
- `get_results_list_by_id()`: Single lookup
- `get_results_by_search_id()`: Get results for specific search
- `update_results_list()`: Accept/reject result
- `delete_results_list()`: Remove result
- `get_results_list_count()`: Count total
- `get_accepted_results()`: Filter by isAccepted=True

---

## PART 5: CLIP MODULE - TEXT-BASED PERSON RETRIEVAL

**File:** `/CLIP/clip_v2.py`

Implements CLIP-based person re-identification for text-to-image retrieval.

### 5.1 CLIP Class

```python
class CLIP:
    def __init__(self, model_name: str = "ViT-B/32", device: str = None)
```

#### Initialization
- Loads OpenAI CLIP model (Vision Transformer B/32)
- Auto-detects GPU availability
- Sets model to eval mode
- Stores preprocessing pipeline

#### Core Methods

1. **encode_images(image_paths: List[str]) -> torch.Tensor**
   - Loads images from file paths
   - Applies CLIP preprocessing (resize, normalize)
   - Encodes to 512-dim vectors
   - L2 normalizes embeddings
   - Returns stacked tensor (N, 512)

2. **encode_person_crop(image: PIL.Image) -> torch.Tensor**
   - Encodes single person crop
   - Used by embedding worker for real-time processing
   - Returns normalized 512-dim vector

3. **encode_text(text_descriptions: List[str]) -> torch.Tensor**
   - Tokenizes text descriptions
   - Encodes to 512-dim vectors
   - L2 normalizes
   - Returns stacked tensor (N, 512)

4. **retrieve(text_query: str, image_paths: List[str], top_k: int = 10, threshold: float = None) -> List[Tuple[str, float]]**
   - Encodes all images
   - Encodes text query
   - Computes cosine similarity (dot product of normalized vectors)
   - Returns top-k matches with similarity scores
   - Optional threshold filtering

5. **batch_retrieve(text_queries: List[str], image_paths: List[str], top_k: int = 10, threshold: float = None) -> dict**
   - Encodes all images once
   - Encodes all queries
   - Computes similarity matrix (n_images × n_queries)
   - Returns dictionary mapping each query to its top-k matches

---

## PART 6: DATA FLOW AND SYSTEM STATES

### 6.1 Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SYSTEM INITIALIZATION                             │
│  1. Streamlit app.py starts                                          │
│  2. Session state initialized (processes, locks, Pid counter)        │
│  3. Database tables created (SQLAlchemy)                             │
│  4. Qdrant collections created (stream, CLIP_embeddings)             │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    USER INTERACTION (Frontend)                       │
│  1. User uploads video files or selects webcams                      │
│  2. User clicks "Start Selected"                                     │
│  3. Streamlit creates multiprocessing processes                      │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│              VIDEO STREAM PROCESSING (MCDPT Module)                  │
│  Process 1: process_stream(video_file_1, cam_id=0, ...)             │
│  Process 2: process_stream(video_file_2, cam_id=1, ...)             │
│  Process 3: process_stream(webcam, cam_id=2, ...)                   │
│                                                                       │
│  For each frame:                                                     │
│  1. Read frame from video/webcam                                     │
│  2. Every 2 frames: Run YOLO detection (GPU, lock-protected)        │
│  3. Filter detections: persons only, confidence >= 0.5              │
│  4. Run DeepSort tracker with OSNet embeddings                       │
│  5. For each tracked person:                                         │
│     - Extract bounding box                                           │
│     - Get persistent track ID                                        │
│     - Draw visualization                                             │
│     - Extract person crop                                            │
│     - Generate OSNet embedding                                       │
│     - Upsert to Qdrant "stream" collection (db_lock-protected)      │
│  6. Display annotated frame                                          │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│          EMBEDDING ENRICHMENT (Embedding Worker Process)             │
│  Process 4: process_embeddings_job(video_paths)                      │
│                                                                       │
│  Every 5 seconds:                                                    │
│  1. Query "stream" collection for new records (batch of 300)        │
│  2. For each record:                                                 │
│     a. Extract metadata (cam_id, coords, frame_count)               │
│     b. Load video frame at frame_count                              │
│     c. Crop person from frame using coordinates                     │
│     d. Run emotion classifier (VIT-Face-Expression)                 │
│     e. Determine if lost (emotion + confidence threshold)           │
│     f. Generate CLIP embedding from crop                            │
│     g. Compare with elderly reference embedding                     │
│     h. Compare with disabled reference embedding                    │
│     i. Upsert to "CLIP_embeddings" with attributes                 │
│  3. Increment batch counter                                          │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│           SQL SYNCHRONIZATION (Sync Worker Process)                  │
│  Process 5: sync_clip_embeddings_to_sql()                            │
│                                                                       │
│  Every 30 seconds:                                                   │
│  1. Query "CLIP_embeddings" collection (batch of 1000)              │
│  2. For each record:                                                 │
│     a. Extract Pid, is_lost, is_elderly, is_disabled               │
│     b. Check if CameraDetectedPerson exists in SQL                 │
│     c. If exists: Update attributes                                 │
│     d. If not exists: Create new CameraDetectedPerson              │
│  3. Commit to SQLite database                                        │
│  4. Increment record counter                                         │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│              FRONTEND DATA DISPLAY & MANAGEMENT                      │
│  1. User navigates to different pages (sidebar)                      │
│  2. Pages query SQL database using CRUD functions                    │
│  3. Display persons, staff, families, search results                 │
│  4. User can:                                                        │
│     - Register new missing persons                                   │
│     - Create search requests                                         │
│     - View detected persons from cameras                             │
│     - Track last seen locations                                      │
│     - Accept/reject search results                                   │
│     - Mark persons as found                                          │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    USER STOPS PROCESSING                             │
│  1. User clicks "Stop All"                                           │
│  2. All processes are terminated gracefully                          │
│  3. Temporary video files are deleted                                │
│  4. Process metadata is cleared from session state                   │
│  5. Data remains in databases (SQL and Qdrant)                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 System States and Transitions

#### State 1: Initialization State
- **Duration**: Application startup
- **Active Components**: Main Streamlit process
- **Data State**: Empty databases, no processes
- **Transition**: User uploads files/selects webcams → Start State

#### State 2: Start State
- **Duration**: User clicks "Start Selected"
- **Active Components**: 
  - Main Streamlit process
  - N stream processing processes
  - 1 embedding worker process
  - 1 SQL sync worker process
- **Data State**: 
  - Qdrant "stream" collection accumulating OSNet embeddings
  - Qdrant "CLIP_embeddings" collection accumulating enriched embeddings
  - SQL database receiving CameraDetectedPerson records
- **Transition**: User clicks "Stop All" → Stop State

#### State 3: Processing State
- **Duration**: Continuous while processes are active
- **Active Components**: All 5 processes running in parallel
- **Data State**:
  - Qdrant "stream": Growing with frame-by-frame embeddings
  - Qdrant "CLIP_embeddings": Growing with enriched embeddings
  - SQL CameraDetectedPerson: Growing with new detections
  - SQL LastSeen: Growing with location records
- **Concurrency Control**:
  - `lock`: Serializes GPU access (YOLO detection, tracker, OSNet)
  - `db_lock`: Serializes Qdrant writes from stream process
  - SQLAlchemy session management: Handles SQL writes

#### State 4: Stop State
- **Duration**: User clicks "Stop All"
- **Active Components**: Main Streamlit process only
- **Data State**: 
  - All processes terminated
  - Qdrant collections frozen (no new writes)
  - SQL database frozen (no new writes)
  - All data persisted
- **Transition**: User clicks "Start Selected" again → Start State

---

### 6.3 Concurrency and Synchronization

#### GPU Lock (`lock`)
- **Protects**: CUDA operations
- **Used by**: 
  - Stream processing (YOLO detection, tracker.process)
  - Embedding worker (CLIP encoding)
- **Mechanism**: multiprocessing.Lock()
- **Granularity**: Coarse (entire GPU operation)

#### Database Lock (`db_lock`)
- **Protects**: Qdrant upsert operations
- **Used by**: Stream processing process
- **Mechanism**: multiprocessing.Lock()
- **Granularity**: Per-batch (multiple points per lock acquisition)

#### Shared Integer (`Pid`)
- **Purpose**: Allocate unique IDs for Qdrant points
- **Type**: multiprocessing.Value('i', 0)
- **Accessed by**: Stream processing process
- **Atomic Operation**: Increment within db_lock critical section

#### SQLAlchemy Session Management
- **Mechanism**: SessionLocal() creates thread-local sessions
- **Isolation**: Each CRUD operation gets fresh session
- **Commit**: Explicit commit after modifications
- **Rollback**: On error, rolls back transaction

---

## PART 7: KEY ALGORITHMS AND MODELS

### 7.1 Person Detection: YOLO RT-DETR-L

**Model**: Real-time Detection Transformer (RT-DETR) with Large backbone
- **Architecture**: Transformer-based object detector
- **Input**: Video frames (variable resolution)
- **Output**: Bounding boxes, class labels, confidence scores
- **Classes**: 80 COCO classes (person is class 0)
- **Confidence Threshold**: 0.5
- **Processing**: Every 2 frames (skip for performance)

### 7.2 Person Re-Identification: OSNet

**Model**: OSNet_x0_25_msmt17 (lightweight)
- **Architecture**: Lightweight CNN for person re-ID
- **Input**: Person crop (256×128 pixels)
- **Output**: 512-dimensional embedding
- **Normalization**: L2 norm (cosine similarity)
- **Purpose**: Generate features for tracking across frames
- **Used by**: DeepSort tracker for data association

### 7.3 Multi-Camera Tracking: DeepSort+

**Algorithm**: Deep SORT with multi-camera extensions
- **Single Camera Tracking**:
  - Kalman filter for motion prediction
  - Hungarian algorithm for data association
  - Cosine distance metric on OSNet embeddings
  - Match threshold: 0.4
  
- **Multi-Camera Tracking**:
  - Trajectory constraints
  - Feature clustering
  - MQTT communication between cameras
  - Cross-camera ID matching

### 7.4 Emotion Detection: ViT-Face-Expression

**Model**: Vision Transformer trained on facial expressions
- **Input**: Person crop (face region)
- **Output**: Emotion probabilities (happy, sad, fear, angry, surprise, neutral)
- **Purpose**: Detect lost/disoriented persons
- **Threshold**: Confidence > 0.6 for sad/fear/angry

### 7.5 Vision-Language Embeddings: CLIP

**Model**: OpenAI CLIP (ViT-B/32)
- **Architecture**: Vision Transformer + Text Transformer
- **Vision Encoder**: ViT-B/32 → 512-dim embeddings
- **Text Encoder**: Transformer → 512-dim embeddings
- **Similarity**: Cosine distance in shared embedding space
- **Applications**:
  - Encode person crops for re-ID
  - Encode text descriptions for search
  - Compare with reference embeddings (elderly, disabled)

### 7.6 Attribute Detection: Similarity-based Classification

**Elderly Detection**:
```python
reference_text = "an elderly person over 60 years old"
elderly_embedding = clip.encode_text([reference_text])
person_embedding = clip.encode_person_crop(crop)
similarity = cosine_similarity(person_embedding, elderly_embedding)
is_elderly = similarity > 0.19
```

**Disability Detection**:
```python
reference_text = "a person in a wheel chair"
disabled_embedding = clip.encode_text([reference_text])
person_embedding = clip.encode_person_crop(crop)
similarity = cosine_similarity(person_embedding, disabled_embedding)
is_disabled = similarity > 0.19
```

---

## PART 8: DATABASE OPERATIONS FLOW

### 8.1 Person Registration Flow

```
User fills Person form
    ↓
validate_person_creation() checks:
  - Duplicate phone number
  - Duplicate ID number
  - Duplicate name + gender
    ↓
If validation passes:
  - Upload images to /Streamlit/uploads/
  - Upload video to /Streamlit/uploads/
  - Create Person record in SQL
  - Link to FamilyMember or SecurityStaff
  - Link to CameraDetectedPerson (if available)
    ↓
Person appears in:
  - 4_Person.py list
  - Search results
  - Statistics
```

### 8.2 Detection-to-Database Flow

```
YOLO detects person in frame
    ↓
DeepSort assigns/maintains track ID
    ↓
OSNet generates embedding
    ↓
Upsert to Qdrant "stream" collection
    ↓
Embedding worker retrieves from "stream"
    ↓
Emotion classifier → is_lost flag
CLIP encoder → person embedding
Compare with elderly/disabled references
    ↓
Upsert to Qdrant "CLIP_embeddings" collection
    ↓
SQL sync worker retrieves from "CLIP_embeddings"
    ↓
Create/Update CameraDetectedPerson in SQL
    ↓
Person appears in:
  - 5_CameraDetectedPerson.py
  - 6_LastSeen.py
  - Search results
```

### 8.3 Search Flow

```
User creates SearchData record
    ↓
User initiates search for missing person
    ↓
System queries CameraDetectedPerson records
    ↓
For each detected person:
  - Compute similarity with missing person's CLIP embedding
  - Create ResultsList entry with match score
    ↓
User views results in 8_ResultsList.py
    ↓
User accepts/rejects matches
    ↓
Update SearchData.isFound flag
    ↓
Update Person.isLost flag if found
```

---

## PART 9: CONFIGURATION AND PARAMETERS

### 9.1 Detection Parameters

**YOLO RT-DETR-L**:
- Confidence threshold: 0.5
- Frame skip: 2 (process every 2nd frame)
- Input size: Variable (model handles)

**DeepSort Tracker**:
- Match threshold: 0.25
- Global match threshold: 0.4
- Time window: 20 frames
- Budget: 20 (max features per track)

**SCT Configuration**:
```python
sct_config = {
    'num_clusters': 10,
    'clust_init_dis_thresh': 0.2,
    'time_window': 7,
    'merge_thresh': 0.6,
    'rectify_thresh': 0.3,
    'stable_time_thresh': 5
}
```

### 9.2 Embedding Parameters

**OSNet**:
- Input size: 256×128
- Output dimension: 512
- Normalization: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**CLIP**:
- Model: ViT-B/32
- Output dimension: 512
- Normalization: L2 norm

**Emotion Classifier**:
- Model: trpakov/vit-face-expression
- Top-k: 1 (best prediction only)
- Confidence threshold: 0.6
- Lost emotions: sad, fear, angry

**Attribute Detection**:
- Elderly threshold: 0.19
- Disabled threshold: 0.19

### 9.3 Processing Parameters

**Embedding Worker**:
- Batch size: 300 records per iteration
- Sleep interval: 5 seconds

**SQL Sync Worker**:
- Batch size: 1000 records per iteration
- Sleep interval: 30 seconds

**Qdrant**:
- URL: http://localhost:6333
- Distance metric: COSINE
- Vector dimension: 512

**SQLite**:
- Database file: lost_persons.db
- Location: /Capstone-AI-SE/
- Check same thread: False (multiprocessing)

---

## PART 10: ERROR HANDLING AND EDGE CASES

### 10.1 Video Processing Edge Cases

1. **Video File Not Found**:
   - cv2.VideoCapture returns False
   - Process exits gracefully
   - No data is written

2. **Frame Extraction Failure**:
   - cap.read() returns (False, None)
   - Loop breaks, process exits
   - Partial data remains in Qdrant

3. **Invalid Coordinates**:
   - Coordinates clamped to frame bounds
   - If clamped box is invalid, crop is skipped
   - No embedding is generated

4. **Webcam Not Available**:
   - cv2.VideoCapture(cam_id) fails
   - Process exits
   - User sees process as "stopped"

### 10.2 Database Edge Cases

1. **Duplicate Phone Number**:
   - Validation catches before insert
   - Error message shows existing person info
   - User can view existing record

2. **Duplicate ID Number**:
   - Validation catches before insert
   - Error message shows existing person info
   - User can view existing record

3. **Duplicate Name + Gender**:
   - Validation catches before insert
   - Error message shows existing person info
   - User can view existing record

4. **Orphaned Records**:
   - CameraDetectedPerson deleted → LastSeen deleted (CASCADE)
   - SearchData deleted → ResultsList deleted (CASCADE)
   - FamilyMember deleted → Person deleted (CASCADE)

### 10.3 Embedding Edge Cases

1. **No Records in Stream Collection**:
   - Embedding worker sleeps and retries
   - No error, just waits for data

2. **Invalid Frame Count**:
   - extract_frame_by_count() returns None
   - Record is skipped
   - No embedding is generated

3. **Invalid Crop Coordinates**:
   - get_person_crop_from_coords() returns None
   - Record is skipped
   - No embedding is generated

4. **Model Loading Failure**:
   - Process crashes
   - User sees process as "stopped"
   - Error logged to console

---

## PART 11: PERFORMANCE CONSIDERATIONS

### 11.1 GPU Utilization

**Lock-Protected Operations**:
- YOLO detection: ~50-100ms per frame
- DeepSort tracking: ~20-50ms per frame
- OSNet embedding: ~10-20ms per person
- CLIP encoding: ~5-10ms per person

**Optimization Strategies**:
- Frame skipping (every 2 frames)
- Batch processing in embedding worker
- GPU memory management with .cpu() transfers

### 11.2 Memory Management

**Qdrant Collections**:
- "stream": Grows unbounded (should implement TTL)
- "CLIP_embeddings": Grows unbounded (should implement TTL)

**SQLite Database**:
- Grows with each detection
- No automatic cleanup
- Manual cleanup via clear_data.py

**Session State**:
- Processes dict: Cleaned on stop
- Process meta dict: Cleaned on stop
- Locks and Pid: Persist across reruns

### 11.3 Scalability Limitations

1. **Single GPU**: All processes share one GPU via lock
2. **Single Qdrant Instance**: Bottleneck for concurrent writes
3. **Single SQLite Database**: Bottleneck for concurrent writes
4. **Streamlit Rerun**: Full page reload on any interaction

---

## PART 12: DEPLOYMENT AND SETUP

### 12.1 Required Services

1. **Qdrant Vector Database**:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Python Environment**:
   - Python 3.8+
   - CUDA 12.1 (for GPU support)
   - Dependencies from requirements.txt

### 12.2 File Structure

```
Capstone-AI-SE/
├── Streamlit/
│   ├── app.py (main entry point)
│   ├── models.py (SQLAlchemy models)
│   ├── database.py (SQLAlchemy config)
│   ├── crud.py (database operations)
│   ├── person_validators.py (validation logic)
│   ├── pages/ (8 management pages)
│   └── uploads/ (user-uploaded media)
├── MCDPT/
│   ├── multiprocessing_mct.py (stream processing)
│   ├── deep_sort_plus/ (tracking algorithm)
│   ├── osnet_x0_25_msmt17.pt (re-ID weights)
│   └── rtdetr-l.pt (detection weights)
├── CLIP/
│   ├── clip_v2.py (CLIP wrapper)
│   └── clip_v1.ipynb (development notebook)
├── Embedding_worker/
│   ├── embedding_worker.py (enrichment worker)
│   ├── helpers.py (utility functions)
│   └── qdrant_local_db/ (local vector DB)
├── lost_persons.db (SQLite database)
├── rtdetr-l.pt (detection weights)
└── requirements.txt (Python dependencies)
```

### 12.3 Startup Command

```bash
streamlit run Streamlit/app.py
```

---

## PART 13: SUMMARY OF SYSTEM FLOW

### Complete User Journey

1. **System Initialization**:
   - User opens Streamlit app
   - Session state initialized
   - Database tables created
   - Qdrant collections created

2. **Video Input**:
   - User uploads video files or selects webcams
   - User clicks "Start Selected"

3. **Stream Processing**:
   - N processes start, each reading video frames
   - Every 2 frames: YOLO detects persons
   - DeepSort tracks persons across frames
   - OSNet generates embeddings
   - Embeddings upserted to Qdrant "stream" collection

4. **Embedding Enrichment**:
   - Embedding worker retrieves records from "stream"
   - Emotion classifier detects lost persons
   - CLIP encoder generates person embeddings
   - Attributes compared with reference embeddings
   - Enriched embeddings upserted to "CLIP_embeddings"

5. **Database Synchronization**:
   - SQL sync worker retrieves from "CLIP_embeddings"
   - Creates/updates CameraDetectedPerson records
   - Data synced to SQLite database

6. **Frontend Display**:
   - User navigates to different pages
   - Pages query SQL database
   - Display detected persons, search results, locations

7. **Person Management**:
   - User registers missing persons
   - User creates search requests
   - System matches detected persons to missing persons
   - User accepts/rejects matches
   - User marks persons as found

8. **Cleanup**:
   - User clicks "Stop All"
   - All processes terminated
   - Temporary files deleted
   - Data persists in databases

---

## PART 14: KEY TECHNICAL INSIGHTS

### 14.1 Why This Architecture?

1. **Multiprocessing**: Allows parallel video processing and GPU utilization
2. **Locks**: Prevent race conditions on shared GPU and database
3. **Two-Stage Embedding**: Stream processing is fast (OSNet), enrichment is thorough (CLIP + emotion)
4. **Qdrant + SQL**: Vector DB for similarity search, SQL for structured queries
5. **Streamlit**: Rapid UI development without frontend framework

### 14.2 Bottlenecks

1. **GPU Lock**: All GPU operations serialized
2. **Qdrant Writes**: Single instance, concurrent writes slow
3. **SQLite Writes**: Single file, concurrent writes slow
4. **Streamlit Rerun**: Full page reload on interaction

### 14.3 Future Improvements

1. **Multi-GPU Support**: Distribute processes across GPUs
2. **Distributed Qdrant**: Cluster for scalability
3. **PostgreSQL**: Replace SQLite for concurrent writes
4. **FastAPI Backend**: Replace Streamlit for better performance
5. **TTL on Qdrant**: Automatic cleanup of old records
6. **Caching**: Cache frequent queries
7. **Indexing**: Add database indexes for faster queries

---

## CONCLUSION

This system represents a sophisticated integration of:
- **Real-time Computer Vision**: YOLO detection, DeepSort tracking
- **Deep Learning**: OSNet re-ID, CLIP vision-language, emotion classification
- **Distributed Processing**: Multiprocessing with synchronization
- **Vector Databases**: Qdrant for similarity search
- **Relational Databases**: SQLite for structured data
- **Web Framework**: Streamlit for rapid UI development

The architecture prioritizes real-time processing and accuracy while maintaining data persistence and user-friendly management interfaces. The system can detect, track, and identify lost persons across multiple camera feeds in real-time, with AI-powered attribute detection (elderly, disabled, lost) and comprehensive database management for security personnel and families.

