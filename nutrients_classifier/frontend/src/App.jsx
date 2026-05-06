import { Routes, Route, Link, useLocation, Navigate } from 'react-router-dom';
import { Camera, LayoutDashboard, Database as DatabaseIcon, Activity, Map, Info } from 'lucide-react';
import Home from './pages/Home';
import Analyze from './pages/Analyze';
import Database from './pages/Database';
import Tracker from './pages/Tracker';
import Analytics from './pages/Analytics';
import Blueprints from './pages/Blueprints';
import Architecture from './pages/Architecture';

import { AuthProvider, AuthContext } from './context/AuthContext';
import { useContext } from 'react';
import Login from './pages/Login';
import Chatbot from './components/Chatbot';

const ProtectedRoute = ({ children }) => {
  const { user } = useContext(AuthContext);
  if (!user) return <Navigate to="/login" />;
  return children;
};

const Navigation = () => {
  const location = useLocation();
  const { user, logout } = useContext(AuthContext);

  const navItems = [
    { path: '/', name: 'Overview', icon: <LayoutDashboard size={18} /> },
    { path: '/analyze', name: 'Analyze', icon: <Camera size={18} /> },
    { path: '/database', name: 'Database', icon: <DatabaseIcon size={18} /> },
    { path: '/tracker', name: 'Tracker', icon: <Activity size={18} /> },
    { path: '/analytics', name: 'Analytics', icon: <Activity size={18} /> },
    { path: '/blueprints', name: 'Blueprints', icon: <Map size={18} /> },
    { path: '/architecture', name: 'Architecture', icon: <Info size={18} /> },
  ];

  return (
    <header className="sticky-header">
      <Link to="/" className="header-logo">🌿 NutriVision AI</Link>
      
      <nav style={{ display: 'flex', gap: '20px' }}>
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link 
              key={item.path} 
              to={item.path}
              style={{
                display: 'flex', alignItems: 'center', gap: '6px',
                padding: '8px 16px', borderRadius: '20px', fontWeight: 600, fontSize: '0.9rem',
                color: isActive ? 'var(--accent-hover)' : 'var(--text-muted)',
                background: isActive ? 'rgba(16, 185, 129, 0.1)' : 'transparent',
                transition: 'all 0.2s ease'
              }}
            >
              {item.icon} {item.name}
            </Link>
          );
        })}
      </nav>

      <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
        <div className="header-badge">
          <span className="pulse-dot"></span> YOLOv8n &bull; CNN
        </div>
        {user ? (
          <button onClick={logout} className="btn secondary" style={{ padding: '6px 12px', fontSize: '0.8rem' }}>Log Out</button>
        ) : (
          <Link to="/login" className="btn primary" style={{ padding: '6px 12px', fontSize: '0.8rem' }}>Log In</Link>
        )}
      </div>
    </header>
  );
};

function App() {
  return (
    <AuthProvider>
      <div className="noise-overlay"></div>
      
      <Navigation />

      <main style={{ width: '90%', maxWidth: '1400px', margin: '0 auto', padding: '2rem 0 5rem' }}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/database" element={<Database />} />
          <Route path="/tracker" element={<ProtectedRoute><Tracker /></ProtectedRoute>} />
          <Route path="/analytics" element={<ProtectedRoute><Analytics /></ProtectedRoute>} />
          <Route path="/blueprints" element={<Blueprints />} />
          <Route path="/architecture" element={<Architecture />} />
        </Routes>
      </main>

      <Chatbot />
    </AuthProvider>
  );
}

export default App;
