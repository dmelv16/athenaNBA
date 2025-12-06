// src/pages/NBAGames.jsx - NBA Games with Live Odds
import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';

const OddsCell = ({ odds, label }) => {
  if (!odds) return <span className="text-slate-600">-</span>;
  
  const american = odds.odds_american;
  const isPositive = american && !american.startsWith('-');
  
  return (
    <div className="text-center">
      {label && <p className="text-[10px] text-slate-500 mb-0.5">{label}</p>}
      <p className={`font-mono font-semibold ${isPositive ? 'text-emerald-400' : 'text-white'}`}>
        {american || '-'}
      </p>
      {odds.line && (
        <p className="text-[10px] text-slate-400">{odds.line > 0 ? '+' : ''}{odds.line}</p>
      )}
    </div>
  );
};

const GameCard = ({ game }) => {
  const { moneyline, spread, total } = game;
  const gameTime = game.start_time ? new Date(game.start_time) : null;
  
  return (
    <div className="card overflow-hidden">
      {/* Header */}
      <div className="bg-slate-800/50 px-4 py-2 flex items-center justify-between border-b border-white/5">
        <span className="text-xs text-slate-400">
          {gameTime ? gameTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : 'TBD'}
        </span>
        <Link 
          to={`/nba/game/${game.game_id}`}
          className="text-xs text-emerald-400 hover:text-emerald-300"
        >
          View Props →
        </Link>
      </div>
      
      {/* Teams and Odds Grid */}
      <div className="p-4">
        {/* Column Headers */}
        <div className="grid grid-cols-5 gap-2 mb-3 text-[10px] text-slate-500 uppercase">
          <div></div>
          <div className="text-center">Moneyline</div>
          <div className="text-center">Spread</div>
          <div className="text-center col-span-2">Total</div>
        </div>
        
        {/* Away Team Row */}
        <div className="grid grid-cols-5 gap-2 items-center py-2 border-b border-white/5">
          <div>
            <p className="font-semibold">{game.away_team}</p>
            <p className="text-[10px] text-slate-500">Away</p>
          </div>
          <OddsCell odds={moneyline?.away} />
          <OddsCell odds={spread?.away} />
          <div className="col-span-2 text-center">
            {total?.over && (
              <div className="flex items-center justify-center gap-2">
                <span className="text-[10px] text-slate-500">O</span>
                <span className="font-mono">{total.over.line}</span>
                <span className={`font-mono text-sm ${total.over.odds_american?.startsWith('-') ? 'text-white' : 'text-emerald-400'}`}>
                  {total.over.odds_american}
                </span>
              </div>
            )}
          </div>
        </div>
        
        {/* Home Team Row */}
        <div className="grid grid-cols-5 gap-2 items-center py-2">
          <div>
            <p className="font-semibold">{game.home_team}</p>
            <p className="text-[10px] text-slate-500">Home</p>
          </div>
          <OddsCell odds={moneyline?.home} />
          <OddsCell odds={spread?.home} />
          <div className="col-span-2 text-center">
            {total?.under && (
              <div className="flex items-center justify-center gap-2">
                <span className="text-[10px] text-slate-500">U</span>
                <span className="font-mono">{total.under.line}</span>
                <span className={`font-mono text-sm ${total.under.odds_american?.startsWith('-') ? 'text-white' : 'text-emerald-400'}`}>
                  {total.under.odds_american}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const NBAGames = () => {
  const [loading, setLoading] = useState(true);
  const [games, setGames] = useState([]);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const today = new Date().toISOString().split('T')[0];
      const res = selectedDate === today
        ? await api.getNBAGameLinesToday()
        : await api.getNBAGameLinesByDate(selectedDate);
      setGames(res.games || []);
    } catch (e) {
      console.error('Failed to load games:', e);
      setGames([]);
    }
    setLoading(false);
  }, [selectedDate]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="spinner w-10 h-10"></div>
      </div>
    );
  }

  return (
    <div className="space-y-5 animate-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <span>🏀</span> NBA Games & Odds
            <span className="badge bg-emerald-500/20 text-emerald-400 text-xs">LIVE</span>
          </h1>
          <p className="text-slate-400 text-sm mt-0.5">
            {games.length} games with DraftKings odds
          </p>
        </div>
        <div className="flex gap-2">
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="input-field"
          />
          <button onClick={loadData} className="btn-primary text-sm">Refresh</button>
        </div>
      </div>

      {/* Quick Navigation */}
      <div className="flex gap-2">
        <Link to="/nba" className="btn-secondary text-sm">
          📊 Player Props
        </Link>
        <Link to="/nba/bets" className="btn-secondary text-sm">
          💰 Bet History
        </Link>
      </div>

      {/* Games Grid */}
      {games.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {games.map((game, idx) => (
            <GameCard key={game.game_id || idx} game={game} />
          ))}
        </div>
      ) : (
        <div className="card p-12 text-center">
          <span className="text-4xl block mb-2">🏀</span>
          <p className="text-slate-400">No games found for this date</p>
          <p className="text-slate-500 text-sm mt-1">
            Make sure the odds scraper has run for today's games
          </p>
        </div>
      )}

      {/* Legend */}
      <div className="card p-4">
        <h3 className="font-semibold mb-3">Understanding the Odds</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-slate-400">
          <div>
            <h4 className="text-white font-medium mb-2">Moneyline</h4>
            <p>Bet on which team wins. Negative odds (-150) means favorite, positive (+130) means underdog.</p>
          </div>
          <div>
            <h4 className="text-white font-medium mb-2">Spread</h4>
            <p>Bet on the margin of victory. -5.5 means the team must win by 6+. +5.5 means they can lose by up to 5.</p>
          </div>
          <div>
            <h4 className="text-white font-medium mb-2">Total (O/U)</h4>
            <p>Bet on combined score of both teams. Over means more than the line, Under means less.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NBAGames;