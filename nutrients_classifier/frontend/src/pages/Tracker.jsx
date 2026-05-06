import React, { useState, useEffect, useContext } from 'react';
import { Calendar, CheckCircle2, ChevronRight, Plus, Loader2 } from 'lucide-react';
import { AuthContext } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

const API_URL = 'http://127.0.0.1:8000';

const Tracker = () => {
  const [meals, setMeals] = useState([]);
  const [loading, setLoading] = useState(true);
  const { user } = useContext(AuthContext);
  const navigate = useNavigate();

  useEffect(() => {
    if (!user) return;
    const fetchMeals = async () => {
      try {
        const res = await fetch(`${API_URL}/api/meals?user_id=${user.id}`);
        if (res.ok) {
          const data = await res.json();
          setMeals(data.today);
        }
      } catch (err) {
        console.error("Failed to fetch meals", err);
      } finally {
        setLoading(false);
      }
    };
    fetchMeals();
  }, [user]);

  // Calculate totals
  const totalCals = meals.reduce((sum, meal) => sum + (meal.calories || 0), 0);
  const totalPro = meals.reduce((sum, meal) => sum + (meal.protein || 0), 0);
  const totalCarbs = meals.reduce((sum, meal) => sum + (meal.carbs || 0), 0);
  const totalFat = meals.reduce((sum, meal) => sum + (meal.fat || 0), 0);

  // Hardcoded goals for prototype
  const GOALS = { cals: 2100, pro: 120, carbs: 250, fat: 70 };

  const getEmoji = (name) => {
    const f = name.toLowerCase();
    if (f.includes('burger')) return '🍔';
    if (f.includes('apple')) return '🍎';
    if (f.includes('chicken')) return '🍗';
    return '🍽️';
  };

  const renderMealGroup = (type, title) => {
    const groupMeals = meals.filter(m => m.meal_type === type);
    const groupCals = groupMeals.reduce((sum, m) => sum + (m.calories || 0), 0);

    return (
      <div className="card glass-card" style={{ padding: 0, overflow: 'hidden', border: groupMeals.length === 0 ? '2px dashed var(--border)' : '1px solid var(--border)', background: groupMeals.length === 0 ? 'transparent' : 'var(--surface)', boxShadow: groupMeals.length === 0 ? 'none' : 'var(--shadow)' }}>
        {groupMeals.length > 0 ? (
          <>
            <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255,255,255,0.4)' }}>
              <h3 className="syne-font" style={{ fontSize: '1.4rem' }}>{title}</h3>
              <span style={{ fontWeight: 700, color: 'var(--text)' }}>{groupCals.toFixed(0)} kcal</span>
            </div>
            <div style={{ padding: '1.5rem' }}>
              {groupMeals.map((m, idx) => (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: idx !== groupMeals.length - 1 ? '1rem' : 0 }}>
                  <span style={{ fontSize: '2rem' }}>{getEmoji(m.food_name)}</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 600, textTransform: 'capitalize' }}>{m.food_name}</div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>{m.grams.toFixed(0)}g &bull; {m.calories.toFixed(0)} kcal</div>
                  </div>
                </div>
              ))}
            </div>
            <div 
              onClick={() => navigate(`/analyze?meal=${type}`)}
              style={{ padding: '1rem 1.5rem', background: 'rgba(16, 185, 129, 0.05)', color: 'var(--accent-hover)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer', transition: 'background 0.2s' }}
              onMouseOver={e => e.currentTarget.style.background = 'rgba(16, 185, 129, 0.15)'}
              onMouseOut={e => e.currentTarget.style.background = 'rgba(16, 185, 129, 0.05)'}
            >
              <Plus size={18} /> Add Food
            </div>
          </>
        ) : (
          <div onClick={() => navigate(`/analyze?meal=${type}`)} style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)', cursor: 'pointer', transition: 'transform 0.2s' }} onMouseOver={e => e.currentTarget.style.transform = 'scale(1.02)'} onMouseOut={e => e.currentTarget.style.transform = 'scale(1)'}>
            <h3 className="syne-font" style={{ fontSize: '1.4rem', marginBottom: '0.5rem', color: 'var(--text)' }}>{title}</h3>
            <p>Nothing logged yet.</p>
            <button className="btn primary" style={{ marginTop: '1rem' }}><Plus size={18} /> Log {title}</button>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="fade-in">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3rem' }}>
        <div>
          <h1 className="syne-font" style={{ fontSize: '3rem', marginBottom: '0.5rem' }}>Daily Tracker</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '1.2rem' }}>Live SQLite Sync &bull; Today</p>
        </div>
        <button className="btn secondary"><Calendar size={20} /> History</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '2rem' }}>
        
        {/* Left Column: Meals */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {loading ? (
             <div style={{ textAlign: 'center', padding: '4rem' }}><Loader2 className="spinner" size={40} style={{ animation: 'spin 1s linear infinite' }} /></div>
          ) : (
             <>
               {renderMealGroup('Breakfast', 'Breakfast')}
               {renderMealGroup('Lunch', 'Lunch')}
               {renderMealGroup('Dinner', 'Dinner')}
             </>
          )}
        </div>

        {/* Right Column: Daily Progress */}
        <div>
          <div className="card glass-card" style={{ position: 'sticky', top: '100px' }}>
            <h3 className="syne-font" style={{ fontSize: '1.4rem', marginBottom: '2rem', textAlign: 'center' }}>Daily Summary</h3>
            
            <div style={{ position: 'relative', width: '200px', height: '200px', margin: '0 auto 2rem' }}>
              <svg viewBox="0 0 36 36" style={{ width: '100%', height: '100%', transform: 'rotate(-90deg)' }}>
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="var(--border)" strokeWidth="3" />
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="var(--accent)" strokeWidth="3" strokeDasharray={`${Math.min(100, (totalCals / GOALS.cals) * 100)}, 100`} />
              </svg>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <span className="syne-font" style={{ fontSize: '2.5rem', fontWeight: 800, lineHeight: 1 }}>{totalCals.toFixed(0)}</span>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>/ {GOALS.cals} kcal</span>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px', fontSize: '0.9rem', fontWeight: 600 }}>
                  <span>Protein</span> <span>{totalPro.toFixed(0)}g / {GOALS.pro}g</span>
                </div>
                <div className="progress-track"><div className="progress-fill" style={{ width: `${Math.min(100, (totalPro / GOALS.pro) * 100)}%`, background: 'var(--protein)' }}></div></div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px', fontSize: '0.9rem', fontWeight: 600 }}>
                  <span>Carbs</span> <span>{totalCarbs.toFixed(0)}g / {GOALS.carbs}g</span>
                </div>
                <div className="progress-track"><div className="progress-fill" style={{ width: `${Math.min(100, (totalCarbs / GOALS.carbs) * 100)}%`, background: 'var(--carbs)' }}></div></div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px', fontSize: '0.9rem', fontWeight: 600 }}>
                  <span>Fat</span> <span>{totalFat.toFixed(0)}g / {GOALS.fat}g</span>
                </div>
                <div className="progress-track"><div className="progress-fill" style={{ width: `${Math.min(100, (totalFat / GOALS.fat) * 100)}%`, background: 'var(--fat)' }}></div></div>
              </div>
            </div>

            <div style={{ marginTop: '2rem', padding: '1rem', background: 'var(--bg)', borderRadius: '12px', display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
              <CheckCircle2 color="var(--accent)" style={{ marginTop: '3px', flexShrink: 0 }} />
              <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>{totalCals > 0 ? "You're on track! Logs are synced to SQLite." : "Your database is connected. Snap a photo to log your first meal!"}</p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default Tracker;
