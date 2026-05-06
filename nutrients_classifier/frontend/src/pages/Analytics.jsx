import React, { useState, useEffect, useContext } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend } from 'recharts';
import { Loader2 } from 'lucide-react';
import { AuthContext } from '../context/AuthContext';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const Analytics = () => {
  const [calData, setCalData] = useState([]);
  const [macroData, setMacroData] = useState([]);
  const [loading, setLoading] = useState(true);
  const { user } = useContext(AuthContext);

  useEffect(() => {
    if (!user) return;
    const fetchHistory = async () => {
      try {
        const res = await fetch(`${API_URL}/api/meals?user_id=${user.id}`);
        if (res.ok) {
          const data = await res.json();
          const history = data.history || [];
          
          // Group by day
          const grouped = {};
          
          // Initialize last 7 days
          for (let i = 6; i >= 0; i--) {
            const d = new Date();
            d.setDate(d.getDate() - i);
            const dateStr = d.toLocaleDateString('en-US', { weekday: 'short' });
            grouped[dateStr] = { day: dateStr, kcal: 0, Protein: 0, Carbs: 0, Fat: 0 };
          }

          history.forEach(meal => {
            const dateStr = new Date(meal.timestamp).toLocaleDateString('en-US', { weekday: 'short' });
            if (grouped[dateStr]) {
              grouped[dateStr].kcal += (meal.calories || 0);
              grouped[dateStr].Protein += (meal.protein || 0);
              grouped[dateStr].Carbs += (meal.carbs || 0);
              grouped[dateStr].Fat += (meal.fat || 0);
            }
          });

          const finalData = Object.values(grouped);
          setCalData(finalData);
          setMacroData(finalData);
        }
      } catch (err) {
        console.error("Failed to fetch analytics", err);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, [user]);

  if (loading) {
    return <div style={{ textAlign: 'center', padding: '10rem' }}><Loader2 className="spinner" size={40} style={{ animation: 'spin 1s linear infinite' }} /></div>;
  }

  return (
    <div className="fade-in">
      <div style={{ marginBottom: '3rem' }}>
        <h1 className="syne-font" style={{ fontSize: '3rem', marginBottom: '1rem' }}>Dietary Analytics</h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '1.2rem' }}>Live SQLite sync showing your nutritional trends over the past 7 days.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '3rem' }}>
        
        {/* Caloric Intake Area Chart */}
        <div className="card glass-card">
          <h3 className="syne-font" style={{ fontSize: '1.5rem', marginBottom: '2rem' }}>Caloric Intake Trend</h3>
          <div style={{ width: '100%', height: 350 }}>
            <ResponsiveContainer>
              <AreaChart data={calData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorKcal" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="var(--accent)" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{fill: 'var(--text-muted)'}} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: 'var(--text-muted)'}} />
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                <Tooltip 
                  contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '12px', backdropFilter: 'blur(10px)' }}
                  itemStyle={{ color: 'var(--accent)', fontWeight: 'bold' }}
                />
                <Area type="monotone" dataKey="kcal" stroke="var(--accent)" strokeWidth={3} fillOpacity={1} fill="url(#colorKcal)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Macro Distribution Bar Chart */}
        <div className="card glass-card">
          <h3 className="syne-font" style={{ fontSize: '1.5rem', marginBottom: '2rem' }}>Macro Breakdown</h3>
          <div style={{ width: '100%', height: 350 }}>
            <ResponsiveContainer>
              <BarChart data={macroData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{fill: 'var(--text-muted)'}} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: 'var(--text-muted)'}} />
                <Tooltip 
                  cursor={{fill: 'var(--bg)'}}
                  contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '12px' }}
                />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px' }} />
                <Bar dataKey="Protein" stackId="a" fill="#3b82f6" radius={[0, 0, 4, 4]} />
                <Bar dataKey="Carbs" stackId="a" fill="#f59e0b" />
                <Bar dataKey="Fat" stackId="a" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
};

export default Analytics;
