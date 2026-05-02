import { useState, useEffect, useRef } from 'react';
import { RefreshCw, Upload, AlertCircle } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
const WS_BASE = import.meta.env.VITE_WS_URL || "ws://localhost:8000/api/ws";

export default function CamerasTab({ t, onImageClick }) {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [cameraImages, setCameraImages] = useState({});  // All cameras' images
  const [loading, setLoading] = useState(false);
  const [indexing, setIndexing] = useState({});
  const [error, setError] = useState(null);
  const [logs, setLogs] = useState([]);
  const pollIntervalRef = useRef(null);
  const wsRef = useRef(null);

  // Load cameras on mount
  useEffect(() => {
    fetchCameras();
    setupWebSocket();
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // Poll all cameras continuously
  useEffect(() => {
    if (cameras.length === 0) return;

    fetchAllCameras();
    pollIntervalRef.current = setInterval(() => {
      fetchAllCameras();
    }, 2000);

    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, [cameras]);

  const setupWebSocket = () => {
    const ws = new WebSocket(WS_BASE + "/capture");
    ws.onopen = () => {
      console.log("WebSocket connected for camera indexing updates");
    };
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      setLogs(prev => [...prev, data]);
      
      // Auto-scroll logs
      setTimeout(() => {
        const logBox = document.getElementById('camera-log-box');
        if (logBox) logBox.scrollTop = logBox.scrollHeight;
      }, 0);
    };
    ws.onerror = () => {
      console.error("WebSocket error");
    };
    wsRef.current = ws;
  };

  const fetchCameras = async () => {
    try {
      const res = await fetch(`${API_BASE}/cameras/list`);
      if (res.ok) {
        const data = await res.json();
        setCameras(data);
        setError(null);
        if (data.length > 0 && !selectedCamera) {
          setSelectedCamera(data[0]);
        }
      } else {
        setError("Failed to load cameras");
      }
    } catch (err) {
      console.error("Error fetching cameras:", err);
      setError("Error connecting to server");
    }
  };

  const fetchAllCameras = async () => {
    setLoading(true);
    const newImages = {};
    
    for (const camera of cameras) {
      try {
        const res = await fetch(`${API_BASE}/cameras/${camera}/images`);
        if (res.ok) {
          const data = await res.json();
          newImages[camera] = data;
        }
      } catch (err) {
        console.error(`Error fetching images for ${camera}:`, err);
      }
    }
    
    setCameraImages(newImages);
    setError(null);
    setLoading(false);
  };

  const handleIndexImage = async (camera, imageFilename) => {
    const key = `${camera}/${imageFilename}`;
    setIndexing(prev => ({ ...prev, [key]: true }));
    
    try {
      const res = await fetch(
        `${API_BASE}/cameras/${camera}/image/${imageFilename}/index`,
        { method: 'POST' }
      );
      if (res.ok) {
        console.log("Index request sent, processing in background...");
      } else {
        setLogs(prev => [...prev, {type: 'error', msg: `Failed to index ${imageFilename}`}]);
      }
    } catch (err) {
      console.error("Error indexing image:", err);
      setLogs(prev => [...prev, {type: 'error', msg: `Error indexing ${imageFilename}: ${err.message}`}]);
    }
    setIndexing(prev => ({ ...prev, [key]: false }));
  };

  const images = selectedCamera ? (cameraImages[selectedCamera] || []) : [];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 250px', gap: '1rem', height: '100%' }}>
      <div className="glass-panel">
        <div style={{display:'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem'}}>
          <h2>Live Cameras</h2>
          <button className="btn btn-secondary" onClick={fetchCameras}>
            <RefreshCw size={16}/> Refresh
          </button>
        </div>

        {error && (
          <div style={{ padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '8px', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#ef4444' }}>
            <AlertCircle size={20} />
            {error}
          </div>
        )}

        {cameras.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '2rem' }}>
            <Upload size={32} style={{ marginBottom: '1rem', opacity: 0.5 }} />
            <p>No cameras found. Create subfolders in <code>/cameras/</code></p>
          </div>
        ) : (
          <>
            <div style={{ marginBottom: '1.5rem' }}>
              <label style={{fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem', display: 'block'}}>
                Select Camera
              </label>
              <select 
                value={selectedCamera || ''} 
                onChange={(e) => setSelectedCamera(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  background: 'rgba(0,0,0,0.2)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '8px',
                  color: 'var(--text-h)',
                  fontSize: '1rem',
                  cursor: 'pointer'
                }}
              >
                {cameras.map(cam => (
                  <option key={cam} value={cam}>{cam} ({(cameraImages[cam] || []).length})</option>
                ))}
              </select>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3>{selectedCamera}: {images.length} image(s)</h3>
              <span style={{fontSize: '0.85rem', color: 'var(--text-muted)'}}>
                {loading ? 'Updating all...' : 'All updated'}
              </span>
            </div>

            {images.length > 0 ? (
              <div className="grid grid-4" style={{ gap: '1rem' }}>
                {images.map((img, i) => {
                  const indexKey = `${selectedCamera}/${img}`;
                  const isIndexing = indexing[indexKey];
                  return (
                    <div key={i} className="image-card" style={{position: 'relative'}}>
                      <img 
                        src={`${API_BASE}/cameras/${selectedCamera}/image/${img}`} 
                        alt={img}
                        loading="lazy"
                        onClick={() => onImageClick(img)}
                        style={{cursor: 'pointer'}}
                      />
                      <div className="image-overlay">
                        <div style={{fontSize: '0.75rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginBottom: '0.5rem'}}>
                          {img}
                        </div>
                        <button 
                          className="btn btn-secondary" 
                          style={{padding: '4px 8px', fontSize: '0.75rem', width: '100%'}}
                          onClick={(e) => { e.stopPropagation(); handleIndexImage(selectedCamera, img); }}
                          disabled={isIndexing}
                        >
                          {isIndexing ? <RefreshCw className="spin" size={12} style={{marginRight: 4}}/> : <Upload size={12} style={{marginRight: 4}}/>}
                          {isIndexing ? 'Indexing' : 'Index'}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                {loading ? 'Loading...' : `No images in ${selectedCamera}`}
              </div>
            )}
          </>
        )}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div className="glass-panel">
          <h3>Indexing Logs</h3>
          <div id="camera-log-box" className="log-box" style={{ height: '300px' }}>
            {logs.map((log, i) => (
              <div key={i} className={`log-entry log-${log.type}`}>
                <span style={{opacity:0.5}}>[CAM]</span> {log.msg}
              </div>
            ))}
          </div>
        </div>

        <div className="glass-panel" style={{ flex: 1, minHeight: '200px' }}>
          <h3>Camera Status</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.9rem' }}>
            {cameras.map(cam => (
              <div key={cam} style={{ 
                padding: '0.75rem', 
                background: 'rgba(0,0,0,0.2)', 
                borderRadius: '6px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <span>{cam}</span>
                <span style={{ 
                  fontSize: '0.8rem', 
                  color: 'var(--text-muted)',
                  background: 'rgba(0,0,0,0.3)',
                  padding: '2px 6px',
                  borderRadius: '4px'
                }}>
                  {(cameraImages[cam] || []).length} img
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
