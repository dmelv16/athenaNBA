// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import NBAPredictions from './pages/NBAPredictions';
import NHLPredictions from './pages/NHLPredictions';
import AccuracyTracker from './pages/AccuracyTracker';
import HistoricalPicks from './pages/HistoricalPicks';
import GameDetail from './pages/GameDetail';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/nba" element={<NBAPredictions />} />
          <Route path="/nba/game/:gameId" element={<GameDetail sport="nba" />} />
          <Route path="/nhl" element={<NHLPredictions />} />
          <Route path="/accuracy" element={<AccuracyTracker />} />
          <Route path="/history" element={<HistoricalPicks />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;