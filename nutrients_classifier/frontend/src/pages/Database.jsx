import React, { useState, useEffect, useContext } from 'react';
import { Search, Info, Loader2, Plus, Check } from 'lucide-react';
import { AuthContext } from '../context/AuthContext';
import { useSearchParams } from 'react-router-dom';

const API_URL = 'http://127.0.0.1:8000';

const Database = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loggingId, setLoggingId] = useState(null);
  
  const { user } = useContext(AuthContext);
  const [searchParams] = useSearchParams();
  const mealType = searchParams.get('meal') || 'Lunch'; // default to lunch if not specified

  useEffect(() => {
    const fetchResults = async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API_URL}/api/database/search?q=${query}`);
        if (res.ok) {
          const data = await res.json();
          setResults(data.results);
        }
      } catch (err) {
        console.error("Failed to fetch database results", err);
      } finally {
        setLoading(false);
      }
    };
    
    // Debounce the search query
    const timeoutId = setTimeout(() => {
      fetchResults();
    }, 300);
    return () => clearTimeout(timeoutId);
  }, [query]);

  const handleLogMeal = async (item, idx) => {
    if (!user) return alert("Please log in to track meals.");
    setLoggingId(idx);
    try {
      await fetch(`${API_URL}/api/meals`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user.id,
          meal_type: mealType,
          food_name: item.food,
          grams: 100, // Default to 100g for database log
          calories: item.cal,
          protein: item.pro,
          carbs: item.carbs || 0,
          fat: item.fat || 0
        })
      });
      setTimeout(() => setLoggingId(null), 2000);
    } catch (err) {
      console.error(err);
      setLoggingId(null);
    }
  };

  return (
    <div className="fade-in">
      <div style={{ marginBottom: '3rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 className="syne-font" style={{ fontSize: '3rem', marginBottom: '1rem' }}>USDA Database</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '1.2rem' }}>
            {searchParams.get('meal') ? `Searching foods to log for ${mealType}.` : 'Explore comprehensive nutritional profiles across thousands of live entries.'}
          </p>
        </div>
      </div>

      <div className="card glass-card" style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '15px' }}>
        <Search color="var(--text-muted)" size={24} />
        <input 
          type="text" 
          placeholder="Search the USDA database (e.g., Apple, Chicken)..." 
          value={query}
          onChange={e => setQuery(e.target.value)}
          style={{ border: 'none', background: 'transparent', outline: 'none', fontSize: '1.2rem', width: '100%', color: 'var(--text)' }}
        />
        {loading && <Loader2 size={24} className="spinner" style={{ margin: 0, animation: 'spin 1s linear infinite' }} />}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1.5rem' }}>
        {results.length > 0 ? results.map((item, idx) => (
          <div key={idx} className="card fade-in" style={{ padding: '1.5rem', display: 'flex', gap: '15px', alignItems: 'center', cursor: 'pointer', transition: 'transform 0.2s', ':hover': { transform: 'translateY(-5px)' } }}>
            <div style={{ fontSize: '2.5rem', background: 'var(--bg)', padding: '10px', borderRadius: '12px', minWidth: '70px', textAlign: 'center' }}>
              {item.icon}
            </div>
            <div style={{ flex: 1 }}>
              <h3 className="syne-font" style={{ fontSize: '1.1rem', marginBottom: '5px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{item.food}</h3>
              <div style={{ display: 'flex', gap: '10px', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                <span style={{ color: 'var(--cal)', fontWeight: 600 }}>{item.cal.toFixed(0)} kcal</span>
                <span>•</span>
                <span>{item.pro.toFixed(1)}g Pro</span>
              </div>
            </div>
            <button 
              onClick={(e) => { e.stopPropagation(); handleLogMeal(item, idx); }}
              className="btn"
              style={{ 
                background: loggingId === idx ? 'var(--accent)' : 'rgba(16, 185, 129, 0.1)', 
                color: loggingId === idx ? 'white' : 'var(--accent)', 
                padding: '8px 12px',
                borderRadius: '8px',
                fontSize: '0.9rem'
              }}
              disabled={loggingId === idx}
            >
              {loggingId === idx ? <Check size={16} /> : <><Plus size={16} /> Log</>}
            </button>
          </div>
        )) : (
          !loading && (
            <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '4rem', color: 'var(--text-muted)' }}>
              <Info size={48} style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
              <p>{query ? `No USDA entries found for "${query}".` : "Enter a search query to load USDA data."}</p>
            </div>
          )
        )}
      </div>
    </div>
  );
};

export default Database;
