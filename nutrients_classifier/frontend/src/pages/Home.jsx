import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Zap, Target, Layers } from 'lucide-react';

const Home = () => {
  return (
    <div className="fade-in" style={{ textAlign: 'center', paddingTop: '4rem' }}>
      <h1 className="hero-title" style={{ fontSize: 'clamp(3rem, 5vw, 6rem)', letterSpacing: '-2px', marginBottom: '1.5rem', lineHeight: 1.1 }}>
        See What's In<br />
        <span style={{ color: 'var(--accent)' }}>Your Food</span>
      </h1>
      
      <p style={{ fontSize: 'clamp(1rem, 1.5vw, 1.5rem)', color: 'var(--text-muted)', maxWidth: '800px', margin: '0 auto 3rem' }}>
        Precision nutrition for the modern era. Experience the future of food analysis with real-time detection and smart dietary insights.
      </p>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '15px', flexWrap: 'wrap', marginBottom: '4rem' }}>
        <div className="stat-pill" style={{ border: '1px solid var(--border)', padding: '10px 24px', borderRadius: '30px', background: 'var(--surface)' }}>
          <strong>500+</strong> Indian Foods
        </div>
        <div className="stat-pill" style={{ border: '1px solid var(--border)', padding: '10px 24px', borderRadius: '30px', background: 'var(--surface)' }}>
          <strong>7k+</strong> USDA Entries
        </div>
        <div className="stat-pill" style={{ border: '1px solid var(--border)', padding: '10px 24px', borderRadius: '30px', background: 'var(--surface)' }}>
          <strong>YOLOv8</strong> Vision Core
        </div>
      </div>

      <Link to="/analyze" className="btn primary" style={{ fontSize: '1.2rem', padding: '1.2rem 3rem', borderRadius: '50px' }}>
        Start Analysis <ArrowRight size={20} />
      </Link>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', marginTop: '6rem', textAlign: 'left' }}>
        <div className="card glass-card">
          <Zap size={32} color="var(--accent)" style={{ marginBottom: '1rem' }} />
          <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Lightning Fast</h3>
          <p style={{ color: 'var(--text-muted)' }}>Powered by a standalone FastAPI backend and a React SPA, enabling sub-second inferences with YOLOv8.</p>
        </div>
        <div className="card glass-card">
          <Target size={32} color="var(--accent2)" style={{ marginBottom: '1rem' }} />
          <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Pinpoint Accuracy</h3>
          <p style={{ color: 'var(--text-muted)' }}>Utilizing a custom CNN trained on 101 food categories specifically tailored for diverse diets.</p>
        </div>
        <div className="card glass-card">
          <Layers size={32} color="var(--accent3)" style={{ marginBottom: '1rem' }} />
          <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Deep Integration</h3>
          <p style={{ color: 'var(--text-muted)' }}>Fuzzy matching against the official USDA Global Nutrition Database to resolve exact macro profiles.</p>
        </div>
      </div>
    </div>
  );
};

export default Home;
