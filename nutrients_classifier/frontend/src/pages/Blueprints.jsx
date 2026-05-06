import React from 'react';
import { Leaf, Flame, Activity, Zap, Play } from 'lucide-react';

const plans = [
  { title: 'Keto Core', icon: <Flame size={40} color="var(--fat)" />, desc: 'High fat, ultra-low carb protocol designed for deep ketosis and sustained energy.', macros: 'P: 20% | C: 5% | F: 75%' },
  { title: 'High Protein Vegan', icon: <Leaf size={40} color="var(--accent)" />, desc: 'Plant-based muscle building. Maximized essential amino acids without the meat.', macros: 'P: 35% | C: 45% | F: 20%' },
  { title: 'Balanced Maintenance', icon: <Activity size={40} color="var(--protein)" />, desc: 'The golden ratio for standard weight maintenance and general wellbeing.', macros: 'P: 30% | C: 40% | F: 30%' },
  { title: 'Athlete Performance', icon: <Zap size={40} color="var(--carbs)" />, desc: 'High carbohydrate fueling for endurance athletes and intense training cycles.', macros: 'P: 20% | C: 60% | F: 20%' },
];

const Blueprints = () => {
  return (
    <div className="fade-in">
      <div style={{ marginBottom: '3rem' }}>
        <h1 className="syne-font" style={{ fontSize: '3rem', marginBottom: '1rem' }}>Dietary Blueprints</h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '1.2rem' }}>Pre-configured macro protocols tailored to your specific physiological goals.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '2rem' }}>
        {plans.map((plan, idx) => (
          <div key={idx} className="card glass-card fade-in" style={{ display: 'flex', flexDirection: 'column', height: '100%', transition: 'transform 0.3s, box-shadow 0.3s', cursor: 'pointer', ':hover': { transform: 'translateY(-10px)', boxShadow: '0 20px 40px rgba(0,0,0,0.1)' } }}>
            <div style={{ marginBottom: '1.5rem', background: 'var(--bg)', display: 'inline-block', padding: '1rem', borderRadius: '16px', alignSelf: 'flex-start' }}>
              {plan.icon}
            </div>
            <h3 className="syne-font" style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>{plan.title}</h3>
            <p style={{ color: 'var(--text-muted)', flex: 1, marginBottom: '2rem' }}>{plan.desc}</p>
            
            <div style={{ padding: '1rem', background: 'var(--bg)', borderRadius: '12px', textAlign: 'center', fontWeight: 600, color: 'var(--text)', marginBottom: '1.5rem', border: '1px dashed var(--border)' }}>
              {plan.macros}
            </div>

            <button className="btn primary" style={{ width: '100%', justifyContent: 'center' }}><Play size={18} /> Activate Blueprint</button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Blueprints;
