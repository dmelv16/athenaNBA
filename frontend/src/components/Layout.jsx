// src/components/Layout.jsx
import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

const Layout = ({ children }) => {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const navItems = [
    { path: '/', label: 'Dashboard', icon: '📊' },
    { path: '/nba', label: 'NBA Predictions', icon: '🏀' },
    { path: '/nhl', label: 'NHL Predictions', icon: '🏒' },
    { path: '/accuracy', label: 'Accuracy Tracker', icon: '🎯' },
    { path: '/history', label: 'Historical Picks', icon: '📜' },
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white flex">
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? 'w-64' : 'w-16'} bg-gray-800 transition-all duration-300 flex flex-col`}>
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          {sidebarOpen && <h1 className="text-xl font-bold text-green-400">Athena Sports</h1>}
          <button 
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-gray-700 rounded"
          >
            {sidebarOpen ? '◀' : '▶'}
          </button>
        </div>
        
        <nav className="flex-1 p-4">
          <ul className="space-y-2">
            {navItems.map(item => (
              <li key={item.path}>
                <Link
                  to={item.path}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                    location.pathname === item.path
                      ? 'bg-green-600 text-white'
                      : 'hover:bg-gray-700 text-gray-300'
                  }`}
                >
                  <span className="text-xl">{item.icon}</span>
                  {sidebarOpen && <span>{item.label}</span>}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
        
        <div className="p-4 border-t border-gray-700">
          {sidebarOpen && (
            <div className="text-sm text-gray-400">
              <p>Last Updated</p>
              <p className="text-green-400">{new Date().toLocaleString()}</p>
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <div className="p-6">
          {children}
        </div>
      </main>
    </div>
  );
};

export default Layout;