import { useState, useEffect, useRef, useCallback } from 'react';
import { Search, BarChart2, Video, Upload, Trash2, StopCircle, RefreshCw, X, Play, Image as ImageIcon, Sun, Moon, Camera } from 'lucide-react';
import './index.css';
import CamerasTab from './CamerasTab';

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
const WS_BASE = import.meta.env.VITE_WS_URL || "ws://localhost:8000/api/ws";

const TRANSLATIONS = {
  en: {
    app_title: "Multi-Modal Image Search",
    tab_search: "Image Search",
    tab_dash: "Dashboard",
    tab_capture: "Capture & Detect",
    tab_cameras: "Live Cameras",
    delete_session: "Delete Session Data",
    search_desc: "Search across the ingested dataset using Text Queries, Image References, or recursively use specific images.",
    text_desc: "Text Description",
    text_placeholder: "e.g., A tall bald man wearing a white thobe...",
    upload_image: "Upload an image to search by visual similarity",
    recursive_active: "Recursive Search Active on ID:",
    clear: "Clear",
    filters: "Metadata Filters",
    cameras: "Select Cameras",
    frames: "Frames (comma separated)",
    threshold: "Similarity Threshold",
    execute_search: "Execute Search",
    search_results: "Search Results",
    no_results: "No results found or waiting for search.",
    query_this: "Query This Image",
    load_more: "Load More Results",
    sys_diag: "System Diagnostics",
    refresh: "Refresh Data",
    qdrant: "Qdrant Vector DB",
    minio: "MinIO Storage",
    online: "Online",
    offline: "Offline",
    status: "Status",
    q_points: "Total Indexed Vectors",
    q_size: "Estimated Vector Storage",
    m_objs: "Total Images",
    m_size: "Total Storage Utilized",
    realtime_head: "Real-time Person Detection",
    realtime_desc: "Upload a video to capture frames and automatically index persons using YOLO.",
    video_upload: "Upload Local Video",
    interval: "Capture Interval (sec)",
    start_proc: "Start Processing",
    stop_proc: "Stop Processing",
    logs: "System Logs",
    captures: "Session Captures"
  },
  ar: {
    app_title: "البحث المتعدد الوسائط للصور",
    tab_search: "البحث عن الصور",
    tab_dash: "لوحة التحكم",
    tab_capture: "الالتقاط والكشف",
    tab_cameras: "الكاميرات المباشرة",
    delete_session: "حذف بيانات الجلسة",
    search_desc: "ابحث في مجموعة البيانات باستخدام النصوص، أو الصور، أو بشكل تكراري.",
    text_desc: "وصف نصي",
    text_placeholder: "مثال: رجل طويل أصلع يرتدي ثوباً أبيض...",
    upload_image: "قم بتحميل صورة للبحث بالتشابه",
    recursive_active: "البحث التكراري نشط على:",
    clear: "مسح",
    filters: "فلاتر البيانات",
    cameras: "اختر الكاميرات",
    frames: "الإطارات (مفصولة بفاصلة)",
    threshold: "عتبة التشابه",
    execute_search: "تنفيذ البحث",
    search_results: "نتائج البحث",
    no_results: "لا توجد نتائج أو بانتظار البحث.",
    query_this: "البحث بهذه الصورة",
    load_more: "عرض المزيد من النتائج",
    sys_diag: "تشخيص النظام",
    refresh: "تحديث البيانات",
    qdrant: "قاعدة بيانات التدفق",
    minio: "تخزين البيانات",
    online: "متصل",
    offline: "غير متصل",
    status: "الحالة",
    q_points: "إجمالي المتجهات",
    q_size: "التخزين المقدر",
    m_objs: "إجمالي الصور",
    m_size: "التخزين المستخدم",
    realtime_head: "الكشف عن الأشخاص المباشر",
    realtime_desc: "قم برفع فيديو لاستخراج الصور تلقائياً وفهرستها.",
    video_upload: "رفع فيديو محلي",
    interval: "فترة الالتقاط (بالثواني)",
    start_proc: "بدء المعالجة",
    stop_proc: "إيقاف المعالجة",
    logs: "سجلات النظام",
    captures: "التقاطات الجلسة"
  }
};

function ImageModal({ src, onClose }) {
  if (!src) return null;
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}><X size={20}/></button>
        <img src={src} alt="Enlarged" />
      </div>
    </div>
  );
}

export default function App() {
  const [lang, setLang] = useState('en');
  const [activeTab, setActiveTab] = useState('search');
  const [selectedImage, setSelectedImage] = useState(null);
  const t = TRANSLATIONS[lang];

  const [theme, setTheme] = useState(() => {
    const savedTheme = localStorage.getItem('app-theme');
    if (savedTheme) return savedTheme;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  });

  useEffect(() => {
    if (lang === 'ar') document.body.classList.add('rtl');
    else document.body.classList.remove('rtl');
  }, [lang]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('app-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  return (
    <div className="app-container">
      <header className="top-header">
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Search size={32} />
          {t.app_title}
        </h1>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <button 
            onClick={toggleTheme} 
            className="btn btn-secondary" 
            style={{ padding: '8px', borderRadius: '50%' }}
            aria-label="Toggle theme"
          >
            {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
          </button>
          <div className="lang-toggle glass-panel" style={{ padding: '4px' }}>
            <button className={lang === 'en' ? 'active' : ''} onClick={() => setLang('en')}>EN</button>
            <button className={lang === 'ar' ? 'active' : ''} onClick={() => setLang('ar')}>AR</button>
          </div>
        </div>
      </header>

      <div className="tabs">
        <button className={`tab ${activeTab === 'search' ? 'active' : ''}`} onClick={() => setActiveTab('search')}>
          <Search size={18} style={{marginRight: 6, verticalAlign: 'text-bottom'}}/> {t.tab_search}
        </button>
        <button className={`tab ${activeTab === 'dash' ? 'active' : ''}`} onClick={() => setActiveTab('dash')}>
          <BarChart2 size={18} style={{marginRight: 6, verticalAlign: 'text-bottom'}}/> {t.tab_dash}
        </button>
        <button className={`tab ${activeTab === 'capture' ? 'active' : ''}`} onClick={() => setActiveTab('capture')}>
          <Video size={18} style={{marginRight: 6, verticalAlign: 'text-bottom'}}/> {t.tab_capture}
        </button>
        <button className={`tab ${activeTab === 'cameras' ? 'active' : ''}`} onClick={() => setActiveTab('cameras')}>
          <Camera size={18} style={{marginRight: 6, verticalAlign: 'text-bottom'}}/> {t.tab_cameras}
        </button>
      </div>

      <main>
        {activeTab === 'search' && <SearchTab t={t} onImageClick={(id) => setSelectedImage(`${API_BASE}/image/${id}`)} />}
        {activeTab === 'dash' && <DashTab t={t} />}
        {activeTab === 'capture' && <CaptureTab t={t} onImageClick={(id) => setSelectedImage(`${API_BASE}/image/${id}`)} />}
        {activeTab === 'cameras' && <CamerasTab t={t} onImageClick={(id) => setSelectedImage(`${API_BASE}/cameras/${id}`)} />}
      </main>

      <ImageModal src={selectedImage} onClose={() => setSelectedImage(null)} />
    </div>
  );
}

function SearchTab({ t, onImageClick }) {
  const [textQuery, setTextQuery] = useState('');
  const [files, setFiles] = useState([]);
  const [recursiveId, setRecursiveId] = useState(null);
  
  const [availCameras, setAvailCameras] = useState([]);
  const [selectedCameras, setSelectedCameras] = useState([]);
  const [frames, setFrames] = useState('');
  const [threshold, setThreshold] = useState(0.0);
  
  const [results, setResults] = useState([]);
  const [limit, setLimit] = useState(20);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/cameras`).then(res => res.json()).then(data => {
      if(Array.isArray(data)) setAvailCameras(data);
    }).catch(()=>{});
  }, []);

  const handleClear = () => {
    setTextQuery('');
    setFiles([]);
    setRecursiveId(null);
    setSelectedCameras([]);
    setFrames('');
    setThreshold(0.0);
    setResults([]);
    setLimit(20);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSearch = async (currentLimit = limit) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('text_query', textQuery);
    if (files.length > 0 && !recursiveId) {
      formData.append('file', files[0]);
    }
    if (recursiveId) formData.append('recursive_id', recursiveId);
    formData.append('cameras', selectedCameras.join(','));
    formData.append('frames', frames);
    formData.append('score_threshold', threshold);
    formData.append('limit', currentLimit);

    try {
      const res = await fetch(`${API_BASE}/search`, { method: 'POST', body: formData });
      if (res.ok) {
        const data = await res.json();
        setResults(data.results || []);
      }
    } catch(err) {
      console.error(err);
    }
    setLoading(false);
  };



  return (
    <div className="grid" style={{ gridTemplateColumns: '1fr 300px' }}>
      <div className="glass-panel">
        <p>{t.search_desc}</p>
        
        <h3 style={{marginTop: '1.5rem'}}>{t.text_desc}</h3>
        <input 
          type="text" 
          value={textQuery} 
          onChange={e => setTextQuery(e.target.value)} 
          placeholder={t.text_placeholder} 
        />

        <div style={{ margin: '1.5rem 0' }}>
          {recursiveId ? (
            <div style={{ padding: '1rem', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '8px', display: 'flex', justifyContent: 'space-between' }}>
              <span><b>{t.recursive_active}</b> {recursiveId}</span>
              <button className="btn btn-secondary" onClick={() => setRecursiveId(null)}><X size={16}/> {t.clear}</button>
            </div>
          ) : (
            <div className="file-drop-area">
              <Upload size={24} style={{ marginBottom: 8, color: '#94a3b8' }}/>
              <div>{files.length > 0 ? `${files.length} images selected` : t.upload_image}</div>
              {files.length > 0 && (
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '10px', justifyContent: 'center' }}>
                  {files.map((f, i) => (
                    <div key={i} style={{ position: 'relative' }}>
                      <img src={URL.createObjectURL(f)} alt="preview" style={{ width: '40px', height: '40px', objectFit: 'cover', borderRadius: '4px' }} />
                      <button 
                        style={{ position: 'absolute', top: '-5px', right: '-5px', background: 'red', color: 'white', borderRadius: '50%', width: '16px', height: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', border: 'none', cursor: 'pointer', fontSize: '10px', padding: 0 }}
                        onClick={(e) => { e.preventDefault(); e.stopPropagation(); setFiles(files.filter((_, index) => index !== i)); }}
                      >
                        <X size={10} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
              <div className="file-input-wrapper" style={{ marginTop: 10 }}>
                <button className="btn btn-secondary">{files.length > 0 ? 'Add Images' : 'Browse Images'}</button>
                <input type="file" accept="image/*" multiple onChange={e => setFiles([...files, ...Array.from(e.target.files)])} ref={fileInputRef}/>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <h3>{t.filters}</h3>
        <div>
          <label style={{fontSize: '0.85rem', color: 'var(--text-muted)'}}>{t.cameras}</label>
          {availCameras.length > 0 ? (
             <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '150px', overflowY: 'auto', background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)', marginTop: '0.5rem' }}>
               {availCameras.map(c => (
                 <label key={c} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.9rem' }}>
                   <input 
                     type="checkbox" 
                     value={c} 
                     checked={selectedCameras.includes(c)}
                     onChange={(e) => {
                       if (e.target.checked) {
                         setSelectedCameras([...selectedCameras, c]);
                       } else {
                         setSelectedCameras(selectedCameras.filter(cam => cam !== c));
                       }
                     }}
                     style={{ accentColor: 'var(--primary-color)', width: '16px', height: '16px', cursor: 'pointer' }}
                   />
                   {c}
                 </label>
               ))}
             </div>
          ) : (
             <input type="text" value={selectedCameras.join(',')} onChange={e => setSelectedCameras(e.target.value.split(','))} placeholder="cam1,cam2" style={{ marginTop: '0.5rem' }} />
          )}
        </div>
        <div>
          <label style={{fontSize: '0.85rem', color: 'var(--text-muted)'}}>{t.frames}</label>
          <input type="text" value={frames} onChange={e => setFrames(e.target.value)} />
        </div>
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <label style={{fontSize: '0.85rem', color: 'var(--text-muted)'}}>{t.threshold}</label>
            <span style={{fontSize: '0.85rem', fontWeight: 'bold'}}>{Math.round(threshold * 100)}%</span>
          </div>
          <input type="range" min="0" max="1" step="0.05" value={threshold} onChange={e => setThreshold(parseFloat(e.target.value))} style={{ width: '100%', marginTop: '0.5rem' }} />
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', marginTop: 'auto' }}>
          <button className="btn btn-secondary" style={{flex: 1}} onClick={handleClear} disabled={loading}>
            <X size={18} />
            {t.clear || "Clear"}
          </button>
          <button className="btn btn-primary" style={{flex: 2}} onClick={() => {setLimit(20); handleSearch(20);}} disabled={loading}>
            {loading ? <RefreshCw className="spin" size={18} /> : <Search size={18} />}
            {t.execute_search}
          </button>
        </div>
      </div>

      <div className="glass-panel" style={{ gridColumn: '1 / -1' }}>
        <h2>{t.search_results}</h2>
        {results.length > 0 ? (
          <>
            <div className="grid grid-4" style={{ marginTop: '1.5rem' }}>
              {results.map((r, i) => (
                <div key={i} className="image-card" onClick={() => onImageClick(r.id)} style={{cursor: 'pointer'}}>
                  <img src={`${API_BASE}/image/${r.id}`} alt="Result" loading="lazy" />
                  <div className="image-overlay">
                    <span style={{fontSize: '0.85rem', fontWeight: 600}}>Score: {r.score.toFixed(3)}</span>
                    <span style={{fontSize: '0.75rem', color: '#cbd5e1'}}>C: {r.cam} | F: {r.frame}</span>
                    <button className="btn btn-secondary" style={{padding: '4px', fontSize: '0.75rem', marginTop: 4}} 
                      onClick={(e) => { e.stopPropagation(); setRecursiveId(r.id); handleSearch(limit); }}>
                      <Search size={12}/> {t.query_this}
                    </button>
                  </div>
                </div>
              ))}
            </div>
            
            {results.length >= limit && (
               <div style={{textAlign: 'center', marginTop: '2rem'}}>
                 <button className="btn btn-secondary" onClick={() => {
                   const newLimit = limit + 20;
                   setLimit(newLimit);
                   handleSearch(newLimit);
                 }} disabled={loading}>
                   {loading ? <RefreshCw className="spin" size={16} /> : <RefreshCw size={16}/>} {t.load_more}
                 </button>
               </div>
            )}
          </>
        ) : (
          <p>{t.no_results}</p>
        )}
      </div>
    </div>
  );
}

function DashTab({ t }) {
  const [stats, setStats] = useState(null);

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      const data = await res.json();
      setStats(data);
    } catch(e) { }
  };

  useEffect(() => { fetchStats(); }, []);

  const formatMB = (bytes) => bytes ? (bytes / (1024*1024)).toFixed(2) + " MB" : "0.00 MB";

  return (
    <div className="glass-panel">
      <div style={{display:'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem'}}>
        <h2>{t.sys_diag}</h2>
        <button className="btn btn-secondary" onClick={fetchStats}><RefreshCw size={16}/> {t.refresh}</button>
      </div>

      {stats ? (
        <div className="grid grid-2">
          <div style={{background: 'rgba(0,0,0,0.2)', padding: '1.5rem', borderRadius: 12, display: 'flex', flexDirection: 'column'}}>
            <h3>{t.qdrant}</h3>
            <p><b>{t.status}:</b> <span style={{color: stats.qdrant_status === 'Online' ? '#10b981' : '#ef4444'}}>● {t.online && stats.qdrant_status === 'Online' ? t.online : t.offline}</span></p>
            <div className="stat-card">
              <div style={{color: 'var(--text-muted)', fontSize: '0.9rem'}}>{t.q_points}</div>
              <div className="stat-val">{(stats.qdrant_points || 0).toLocaleString()}</div>
            </div>
            <div className="stat-card" style={{marginBottom: '1rem'}}>
              <div style={{color: 'var(--text-muted)', fontSize: '0.9rem'}}>{t.q_size}</div>
              <div className="stat-val">{formatMB(stats.qdrant_est_bytes)}</div>
            </div>
            <button className="btn btn-secondary" style={{marginTop: 'auto'}} onClick={() => window.open('http://localhost:6333/dashboard', '_blank')}>
              Open Qdrant UI
            </button>
          </div>
          <div style={{background: 'rgba(0,0,0,0.2)', padding: '1.5rem', borderRadius: 12, display: 'flex', flexDirection: 'column'}}>
            <h3>{t.minio}</h3>
            <p><b>{t.status}:</b> <span style={{color: stats.minio_status === 'Online' ? '#10b981' : '#ef4444'}}>● {t.online && stats.minio_status === 'Online' ? t.online : t.offline}</span></p>
            <div className="stat-card">
              <div style={{color: 'var(--text-muted)', fontSize: '0.9rem'}}>{t.m_objs}</div>
              <div className="stat-val">{(stats.minio_objects || 0).toLocaleString()}</div>
            </div>
            <div className="stat-card" style={{marginBottom: '1rem'}}>
              <div style={{color: 'var(--text-muted)', fontSize: '0.9rem'}}>{t.m_size}</div>
              <div className="stat-val">{formatMB(stats.minio_size_bytes)}</div>
            </div>
            <button className="btn btn-secondary" style={{marginTop: 'auto'}} onClick={() => window.open('http://localhost:9001', '_blank')}>
              Open MinIO Console
            </button>
          </div>
        </div>
      ) : (
        <p>Loading stats...</p>
      )}
    </div>
  );
}

function CaptureTab({ t, onImageClick }) {
  const [vidFile, setVidFile] = useState(null);
  const [interval, setIntervalVal] = useState(5.0);
  const [confidence, setConfidence] = useState(0.5);
  const [active, setActive] = useState(false);
  const [logs, setLogs] = useState([]);
  const [captures, setCaptures] = useState([]);
  const wsRef = useRef(null);

  useEffect(() => {
    if (active && !wsRef.current) {
      const ws = new WebSocket(WS_BASE + "/capture");
      ws.onopen = () => {
         setLogs(prev => [...prev, {type: 'success', msg: 'WebSocket connected successfully.'}]);
      };
      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.type === 'crop') {
          setCaptures(prev => [data, ...prev]); 
        } else {
          setLogs(prev => [...prev, data]);
          const logBox = document.getElementById('log-box-inner');
          if(logBox) logBox.scrollTop = logBox.scrollHeight;
        }
      };
      ws.onerror = () => {
         setLogs(prev => [...prev, {type: 'error', msg: 'WebSocket connection failed.'}]);
      };
      wsRef.current = ws;
    } else if (!active && wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, [active]);

  const handleStart = async () => {
    if (!vidFile) return alert("Please select a video file!");
    const formData = new FormData();
    formData.append('file', vidFile);
    formData.append('interval', interval);
    formData.append('conf', confidence);

    setActive(true);
    setLogs([{type:'info', msg:'Starting capture process. Uploading video... Please stand by.'}]);
    setCaptures([]);

    try {
      const res = await fetch(`${API_BASE}/capture/start`, { method: 'POST', body: formData });
      if(!res.ok) {
         setLogs(prev => [...prev, {type:'error', msg:'Error: Server rejected the request.'}]);
         setActive(false);
      }
    } catch(e) { 
      setLogs(prev => [...prev, {type:'error', msg:'Failed to communicate with API.'}]);
      setActive(false);
    }
  };

  const handleStop = async () => {
    try {
       await fetch(`${API_BASE}/capture/stop`, { method: 'POST' });
    } catch (e) {
       console.error("Stop failed", e);
    }
    setActive(false);
    setLogs(prev => [...prev, {type: 'info', msg: 'Capture process stopped by user. Displaying all session captures.'}]);
  };

  return (
    <div className="grid grid-2">
      <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div>
          <h2>{t.realtime_head}</h2>
          <p>{t.realtime_desc}</p>
        </div>

        <div className="file-drop-area">
          <Upload size={24} style={{ marginBottom: 8, color: '#94a3b8' }}/>
          <div>{vidFile ? vidFile.name : t.video_upload}</div>
          <div className="file-input-wrapper" style={{ marginTop: 10 }}>
            <button className="btn btn-secondary">{vidFile ? 'Change Video' : 'Browse Files'}</button>
            <input type="file" accept="video/*" onChange={e => setVidFile(e.target.files[0])} />
          </div>
        </div>

        <div>
           <div style={{ display: 'flex', justifyContent: 'space-between' }}>
             <label style={{fontSize: '0.85rem', color: 'var(--text-muted)'}}>{t.interval}</label>
             <span style={{fontSize: '0.85rem', fontWeight: 'bold'}}>{interval}s</span>
           </div>
           <input type="range" min="1" max="10" step="1" value={interval} onChange={e => setIntervalVal(parseFloat(e.target.value))} style={{ width: '100%', marginTop: '0.5rem' }} />
        </div>

        <div>
           <div style={{ display: 'flex', justifyContent: 'space-between' }}>
             <label style={{fontSize: '0.85rem', color: 'var(--text-muted)'}}>Confidence Threshold</label>
             <span style={{fontSize: '0.85rem', fontWeight: 'bold'}}>{Math.round(confidence * 100)}%</span>
           </div>
           <input type="range" min="0" max="1" step="0.05" value={confidence} onChange={e => setConfidence(parseFloat(e.target.value))} style={{ width: '100%', marginTop: '0.5rem' }} />
        </div>

        <div style={{ marginTop: 'auto', display: 'flex', gap: '1rem' }}>
          {!active ? (
            <button className="btn btn-primary" style={{flex: 1}} onClick={handleStart}>
              <Play size={18}/> {t.start_proc}
            </button>
          ) : (
            <button className="btn btn-danger" style={{flex: 1}} onClick={handleStop}>
              <StopCircle size={18}/> {t.stop_proc}
            </button>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div className="glass-panel">
          <h3>{t.logs}</h3>
          <div id="log-box-inner" className="log-box">
             {logs.map((L, i) => (
               <div key={i} className={`log-entry log-${L.type}`}>
                 <span style={{opacity:0.5}}>[SYS]</span> {L.msg}
               </div>
             ))}
          </div>
        </div>

        <div className="glass-panel" style={{ flex: 1, minHeight: '300px' }}>
          <h3>{t.captures}</h3>
          <div className="grid grid-4" style={{ gap: '0.5rem' }}>
            {captures.map((c, i) => (
              <div key={c.id || i} className="image-card" style={{ borderRadius: 8, cursor: 'pointer' }} onClick={() => onImageClick(c.id)}>
                <img src={`${API_BASE}/image/${c.id}`} alt="Crop" />
                <div className="image-overlay" style={{padding: '4px'}}>
                  <span style={{fontSize: '10px'}}>{c.sec}s • P{c.person}</span>
                </div>
              </div>
            ))}
          </div>
          {captures.length === 0 && <p style={{textAlign:'center', marginTop:'2rem'}}>Waiting for YOLO detections...</p>}
        </div>
      </div>
    </div>
  );
}
