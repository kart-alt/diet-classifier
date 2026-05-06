import React from 'react';
import { Server, Database, BrainCircuit, MonitorSmartphone } from 'lucide-react';

const Architecture = () => {
  return (
    <div className="fade-in">
      <div style={{ marginBottom: '3rem' }}>
        <h1 className="syne-font" style={{ fontSize: '3rem', marginBottom: '1rem' }}>System Architecture</h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '1.2rem' }}>How NutriVision AI processes and analyzes your meals.</p>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        
        <div className="card glass-card fade-in" style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
          <div style={{ background: 'var(--accent)', padding: '1.5rem', borderRadius: '16px', color: 'white' }}>
            <MonitorSmartphone size={48} />
          </div>
          <div>
            <h3 className="syne-font" style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>1. Frontend React SPA</h3>
            <p style={{ color: 'var(--text-muted)' }}>A lightning-fast, multi-page Single Page Application built with Vite and React. Handles all UI rendering, canvas bounding box drawing, and user interactions natively in the browser for zero-latency feedback.</p>
          </div>
        </div>

        <div className="card glass-card fade-in" style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
          <div style={{ background: 'var(--protein)', padding: '1.5rem', borderRadius: '16px', color: 'white' }}>
            <Server size={48} />
          </div>
          <div>
            <h3 className="syne-font" style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>2. FastAPI Backend Engine</h3>
            <p style={{ color: 'var(--text-muted)' }}>The core logic layer. Written in Python, it receives image blobs via REST endpoints, marshals data between the neural networks, and structures the final nutritional output payload.</p>
          </div>
        </div>

        <div className="card glass-card fade-in" style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
          <div style={{ background: 'var(--carbs)', padding: '1.5rem', borderRadius: '16px', color: 'white' }}>
            <BrainCircuit size={48} />
          </div>
          <div>
            <h3 className="syne-font" style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>3. Deep Learning Pipeline</h3>
            <p style={{ color: 'var(--text-muted)' }}>A dual-model architecture: YOLOv8 rapidly localizes food items within the frame, which are then cropped and fed into a custom Keras Convolutional Neural Network (trained on 101 classes) for pinpoint classification.</p>
          </div>
        </div>

        <div className="card glass-card fade-in" style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
          <div style={{ background: 'var(--fat)', padding: '1.5rem', borderRadius: '16px', color: 'white' }}>
            <Database size={48} />
          </div>
          <div>
            <h3 className="syne-font" style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>4. USDA Nutrition Resolution</h3>
            <p style={{ color: 'var(--text-muted)' }}>Detected classes are run through a Levenshtein-distance fuzzy matcher against a localized SQLite/JSON instance of the Global USDA database to extract exact gram-for-gram macro and micro nutrient profiles.</p>
          </div>
        </div>

      </div>
    </div>
  );
};

export default Architecture;
