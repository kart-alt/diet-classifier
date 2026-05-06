import React, { useState, useRef, useCallback, useContext } from 'react';
import { UploadCloud, Loader2, AlertCircle, Camera, ScanBarcode } from 'lucide-react';
import Webcam from 'react-webcam';
import BarcodeScanner from '../components/BarcodeScanner';
import { AuthContext } from '../context/AuthContext';
import { useSearchParams } from 'react-router-dom';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const Analyze = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('macros');
  const [showWebcam, setShowWebcam] = useState(false);
  const [showBarcodeScanner, setShowBarcodeScanner] = useState(false);

  const { user } = useContext(AuthContext);
  const [searchParams] = useSearchParams();
  const urlMealType = searchParams.get('meal');

  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const webcamRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResults(null);
      setError(null);
      setShowWebcam(false);
    }
  };

  const captureWebcam = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      setPreview(imageSrc);
      fetch(imageSrc)
        .then(res => res.blob())
        .then(blob => {
          const newFile = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
          setFile(newFile);
        });
      setShowWebcam(false);
      setResults(null);
      setError(null);
    }
  }, [webcamRef]);

  const handleBarcodeScanned = async (barcode) => {
    setShowBarcodeScanner(false);
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`https://world.openfoodfacts.org/api/v0/product/${barcode}.json`);
      const data = await res.json();
      if (data.status === 1) {
        const p = data.product;
        const name = p.product_name || 'Unknown Product';
        const cals = p.nutriments?.['energy-kcal_100g'] || 0;
        const pro = p.nutriments?.proteins_100g || 0;
        const carbs = p.nutriments?.carbohydrates_100g || 0;
        const fat = p.nutriments?.fat_100g || 0;
        
        // Mocking 100g portion for barcode scan
        const mockResults = {
          detections: [{ food: name, grams: 100, conf: 100, box: [0,0,0,0], calories: cals, emoji: '🏷️' }],
          nutrients: {
            "Calories (kcal)": cals, "Protein (g)": pro, "Carbs (g)": carbs, "Fat (g)": fat,
            "Fiber (g)": p.nutriments?.fiber_100g || 0, "Sugar (g)": p.nutriments?.sugars_100g || 0,
            "Sodium (mg)": p.nutriments?.sodium_100g || 0, "Vitamin C (mg)": 0, "Calcium (mg)": 0, "Iron (mg)": 0
          },
          insights: [{ title: "Barcode Match", desc: `Found ${name} via OpenFoodFacts database.`, icon: "✅", type: "tip" }],
          image_size: [0,0]
        };
        
        setResults(mockResults);
        setPreview(p.image_front_url || null); // Show product image if available
        await saveMealToDatabase(mockResults);
      } else {
        setError("Barcode not found in database.");
      }
    } catch (err) {
      setError("Failed to look up barcode.");
    } finally {
      setLoading(false);
    }
  };

  const drawBoundingBoxes = (imgUrl, detections) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      if (detections && detections.length > 0 && detections[0].box[2] > 0) {
        ctx.lineWidth = Math.max(3, img.width * 0.005);
        ctx.strokeStyle = '#10b981';
        ctx.font = `bold ${Math.max(16, img.width * 0.02)}px Syne`;
        
        detections.forEach(d => {
          const [x1, y1, x2, y2] = d.box;
          if(x2 <= 0) return; // skip if it's a barcode mock with [0,0,0,0]
          
          ctx.beginPath();
          ctx.rect(x1, y1, x2 - x1, y2 - y1);
          ctx.stroke();

          const text = `${d.food} ${d.grams.toFixed(0)}g`;
          const textMetrics = ctx.measureText(text);
          const pad = 8;
          const th = parseInt(ctx.font, 10);
          
          ctx.fillStyle = 'rgba(255,255,255,0.95)';
          ctx.fillRect(x1, y1, textMetrics.width + pad * 2, th + pad * 2);
          
          ctx.fillStyle = '#059669';
          ctx.fillText(text, x1 + pad, y1 + th + pad - 4);
        });
      }
    };
    img.src = imgUrl;
  };

  const saveMealToDatabase = async (data) => {
    if (!user) return; // Don't save if not logged in
    
    let dynamicMealType = urlMealType;
    if (!dynamicMealType) {
      // Dynamically determine meal type based on current time
      const hour = new Date().getHours();
      dynamicMealType = 'Lunch';
      if (hour < 11) dynamicMealType = 'Breakfast';
      else if (hour >= 16) dynamicMealType = 'Dinner';
    }

    try {
      for (const d of data.detections) {
        await fetch(`${API_URL}/api/meals`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: user.id,
            meal_type: dynamicMealType, 
            food_name: d.food,
            grams: d.grams,
            calories: d.calories,
            protein: 0, carbs: 0, fat: 0 
          })
        });
      }
    } catch (err) {
      console.error("Failed to save to database:", err);
    }
  };

  const analyzeImage = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Failed to analyze image. Ensure the backend server is running.');
      
      const data = await res.json();
      setResults(data);
      drawBoundingBoxes(preview, data.detections);
      
      await saveMealToDatabase(data);
    } catch (err) {
      setError(err.message);
      drawBoundingBoxes(preview, []); 
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fade-in">
      {showBarcodeScanner && <BarcodeScanner onResult={handleBarcodeScanned} onClose={() => setShowBarcodeScanner(false)} />}
      
      <div style={{ marginBottom: '3rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 className="syne-font" style={{ fontSize: '3rem', marginBottom: '1rem' }}>Dietary Analyzer</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '1.2rem' }}>Upload or snap a photo of your meal to receive instant breakdowns.</p>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button className="btn secondary" onClick={() => setShowBarcodeScanner(true)}>
            <ScanBarcode size={20} /> Scan Barcode
          </button>
          <button className="btn secondary" onClick={() => setShowWebcam(!showWebcam)}>
            <Camera size={20} /> {showWebcam ? "Close Camera" : "Use Camera"}
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        
        {/* Left Column: Image Area */}
        <div>
          <div 
            className="card" 
            style={{ 
              border: `2px dashed ${preview || showWebcam ? 'var(--border)' : 'var(--accent)'}`,
              padding: preview || showWebcam ? '1rem' : '4rem 2rem',
              textAlign: 'center',
              cursor: (!preview && !showWebcam) ? 'pointer' : 'default',
              minHeight: '400px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              position: 'relative'
            }}
            onClick={() => (!preview && !showWebcam) && fileInputRef.current?.click()}
          >
            <input type="file" hidden ref={fileInputRef} accept="image/*" onChange={handleFileChange} />
            
            {showWebcam ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  style={{ width: '100%', borderRadius: '12px' }}
                />
                <button className="btn primary" onClick={captureWebcam}>Snap Photo</button>
              </div>
            ) : !preview ? (
              <>
                <UploadCloud size={64} color="var(--accent)" style={{ margin: '0 auto 1rem' }} />
                <h3 className="syne-font" style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Upload Meal</h3>
                <p style={{ color: 'var(--text-muted)' }}>Drag & drop or click to browse</p>
              </>
            ) : (
              <canvas ref={canvasRef} style={{ width: '100%', borderRadius: '12px', background: '#000' }} />
            )}
          </div>

          <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
            <button 
              className="btn primary" 
              style={{ flex: 1, justifyContent: 'center' }} 
              disabled={!preview || loading}
              onClick={analyzeImage}
            >
              {loading ? <><Loader2 className="spinner" size={20} style={{ animation: 'spin 1s linear infinite', margin: 0 }} /> Analyzing...</> : 'Analyze Image'}
            </button>
            <button 
              className="btn secondary" 
              style={{ flex: 1, justifyContent: 'center' }} 
              disabled={!preview || loading}
              onClick={() => { setFile(null); setPreview(null); setResults(null); }}
            >
              Clear
            </button>
          </div>

          {error && (
            <div style={{ padding: '1rem', background: '#fee2e2', color: '#b91c1c', borderRadius: '12px', marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <AlertCircle size={20} /> {error}
            </div>
          )}
        </div>

        {/* Right Column: Results */}
        <div>
          {results ? (
            <div className="fade-in">
              <div className="card glass-card" style={{ marginBottom: '2rem' }}>
                <h3 className="syne-font" style={{ fontSize: '1.5rem', marginBottom: '1.5rem' }}>Detected Items</h3>
                {results.detections.length === 0 ? <p>No items detected.</p> : results.detections.map((d, idx) => (
                  <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '15px', padding: '1rem', background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: '12px', marginBottom: '10px' }}>
                    <div style={{ fontSize: '2rem', background: 'white', width: '50px', height: '50px', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '10px' }}>{d.emoji}</div>
                    <div style={{ flex: 1 }}>
                      <div className="syne-font" style={{ fontSize: '1.2rem' }}>{d.food}</div>
                      <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>{d.grams.toFixed(0)}g &bull; {d.calories.toFixed(0)} kcal</div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '0.8rem', color: 'var(--accent)', fontWeight: 700 }}>{d.conf.toFixed(1)}% Match</div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="card glass-card" style={{ padding: 0, overflow: 'hidden' }}>
                <div style={{ display: 'flex', borderBottom: '1px solid var(--border)' }}>
                  {['macros', 'micros', 'insights'].map(tab => (
                    <button 
                      key={tab}
                      onClick={() => setActiveTab(tab)}
                      style={{ flex: 1, padding: '1rem', border: 'none', background: activeTab === tab ? '#fff' : 'transparent', borderBottom: `3px solid ${activeTab === tab ? 'var(--accent)' : 'transparent'}`, fontWeight: 700, cursor: 'pointer', textTransform: 'capitalize' }}
                    >
                      {tab}
                    </button>
                  ))}
                </div>

                <div style={{ padding: '2rem' }}>
                  {activeTab === 'macros' && (
                    <div className="fade-in" style={{ textAlign: 'center' }}>
                      <div style={{ marginBottom: '2rem' }}>
                        <span className="syne-font" style={{ fontSize: '4rem', color: 'var(--text)' }}>{results.nutrients['Calories (kcal)'].toFixed(0)}</span>
                        <span style={{ color: 'var(--text-muted)', fontSize: '1.2rem', marginLeft: '5px' }}>kcal</span>
                      </div>
                      <div style={{ display: 'flex', height: '16px', borderRadius: '8px', overflow: 'hidden', marginBottom: '2rem', boxShadow: 'inset 0 2px 5px rgba(0,0,0,0.05)' }}>
                        <div style={{ width: `${(results.nutrients['Protein (g)'] / (results.nutrients['Protein (g)'] + results.nutrients['Carbs (g)'] + results.nutrients['Fat (g)']) || 1) * 100}%`, background: 'var(--protein)' }}></div>
                        <div style={{ width: `${(results.nutrients['Carbs (g)'] / (results.nutrients['Protein (g)'] + results.nutrients['Carbs (g)'] + results.nutrients['Fat (g)']) || 1) * 100}%`, background: 'var(--carbs)' }}></div>
                        <div style={{ width: `${(results.nutrients['Fat (g)'] / (results.nutrients['Protein (g)'] + results.nutrients['Carbs (g)'] + results.nutrients['Fat (g)']) || 1) * 100}%`, background: 'var(--fat)' }}></div>
                      </div>
                      
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
                        <div style={{ background: '#fff', padding: '1rem', borderRadius: '12px', border: '1px solid var(--border)', borderBottom: '4px solid var(--protein)' }}>
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Protein</div>
                          <div className="syne-font" style={{ fontSize: '1.3rem' }}>{results.nutrients['Protein (g)'].toFixed(1)}g</div>
                        </div>
                        <div style={{ background: '#fff', padding: '1rem', borderRadius: '12px', border: '1px solid var(--border)', borderBottom: '4px solid var(--carbs)' }}>
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Carbs</div>
                          <div className="syne-font" style={{ fontSize: '1.3rem' }}>{results.nutrients['Carbs (g)'].toFixed(1)}g</div>
                        </div>
                        <div style={{ background: '#fff', padding: '1rem', borderRadius: '12px', border: '1px solid var(--border)', borderBottom: '4px solid var(--fat)' }}>
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Fat</div>
                          <div className="syne-font" style={{ fontSize: '1.3rem' }}>{results.nutrients['Fat (g)'].toFixed(1)}g</div>
                        </div>
                        <div style={{ background: '#fff', padding: '1rem', borderRadius: '12px', border: '1px solid var(--border)', borderBottom: '4px solid var(--fiber)' }}>
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Fiber</div>
                          <div className="syne-font" style={{ fontSize: '1.3rem' }}>{results.nutrients['Fiber (g)'].toFixed(1)}g</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === 'micros' && (
                    <div className="fade-in">
                      {[['Vitamin C (mg)', 'var(--accent2)', 90, 'mg'], ['Calcium (mg)', 'var(--accent3)', 1000, 'mg'], ['Iron (mg)', 'var(--text-muted)', 18, 'mg'], ['Sugar (g)', 'var(--fat)', 50, 'g']].map(v => (
                        <div key={v[0]} style={{ display: 'flex', alignItems: 'center', marginBottom: '1.5rem' }}>
                          <div style={{ width: '130px', fontWeight: 600 }}>{v[0].split(' ')[0]}</div>
                          <div style={{ flex: 1, margin: '0 20px', background: 'var(--bg)', borderRadius: '10px', height: '8px', overflow: 'hidden' }}>
                            <div style={{ width: `${Math.min(100, (results.nutrients[v[0]] / v[2]) * 100)}%`, background: v[1], height: '100%', transition: 'width 1s' }}></div>
                          </div>
                          <div style={{ width: '70px', textAlign: 'right', color: 'var(--text-muted)' }}>{results.nutrients[v[0]].toFixed(1)}{v[3]}</div>
                        </div>
                      ))}
                    </div>
                  )}

                  {activeTab === 'insights' && (
                    <div className="fade-in">
                      {results.insights.length === 0 ? <p>No specific insights.</p> : results.insights.map((ins, idx) => (
                        <div key={idx} style={{ display: 'flex', gap: '15px', background: '#fff', border: '1px solid var(--border)', borderLeft: `4px solid ${ins.type === 'alt' ? 'var(--accent)' : 'var(--text-muted)'}`, padding: '1.5rem', borderRadius: '0 12px 12px 0', marginBottom: '1rem' }}>
                          <div style={{ fontSize: '1.8rem' }}>{ins.icon}</div>
                          <div>
                            <div style={{ fontWeight: 700, marginBottom: '5px', fontSize: '1.1rem' }}>{ins.title}</div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>{ins.desc}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="card glass-card" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
              {loading ? (
                <div style={{ textAlign: 'center' }}>
                  <Loader2 size={40} className="spinner" style={{ animation: 'spin 1s linear infinite', margin: '0 auto 1rem' }} />
                  <p>Running models via FastAPI...</p>
                </div>
              ) : (
                <p>Upload an image to see analysis.</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Analyze;
