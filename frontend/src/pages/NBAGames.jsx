// src/pages/NBAGames.jsx - NBA Games with Model Predictions vs DK Lines
import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';

const EdgeBadge = ({ edge, type }) => {
  if (!edge || !edge.recommendation || edge.recommendation === 'no_bet') {
    return <span className="text-slate-500 text-xs">No edge</span>;
  }
  
  const conf = edge.confidence || 'low';
  const colorClass = conf === 'high' 
    ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
    : conf === 'medium'
    ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
    : 'bg-slate-500/20 text-slate-400 border-slate-500/30';
  
  const edgePts = edge.edge_points || 0;
  
  return (
    <span className={`badge border ${colorClass}`}>
      {(edge.recommendation || '').toUpperCase()} ({edgePts > 0 ? '+' : ''}{edgePts} pts)
    </span>
  );
};

const ComparisonRow = ({ label, predicted, actual, edge, unit = '', homeTeam }) => {
  // For spread: convert model's prediction to DK convention for display
  // Model: positive = home wins, DK: negative = home favored
  const isSpread = label === 'Spread';
  
  // Model prediction in DK terms (flip sign for spread)
  const modelInDkTerms = isSpread && predicted != null ? -predicted : predicted;
  
  // Calculate edge properly
  const edgePts = edge?.edge_points;
  
  return (
    <div className="grid grid-cols-4 gap-2 py-2 border-b border-white/5 last:border-0">
      <div className="text-slate-400 text-sm">{label}</div>
      <div className="text-center">
        <span className="font-mono font-semibold text-emerald-400">
          {modelInDkTerms != null ? `${modelInDkTerms > 0 ? '+' : ''}${modelInDkTerms.toFixed(1)}${unit}` : '-'}
        </span>
        <p className="text-[10px] text-slate-500">Model {isSpread && homeTeam ? `(${homeTeam})` : ''}</p>
      </div>
      <div className="text-center">
        <span className="font-mono text-slate-300">
          {actual != null ? `${actual > 0 ? '+' : ''}${actual}${unit}` : '-'}
        </span>
        <p className="text-[10px] text-slate-500">DraftKings</p>
      </div>
      <div className="text-center">
        {edgePts != null ? (
          <span className={`font-mono font-semibold ${Math.abs(edgePts) > 2 ? 'text-amber-400' : 'text-slate-400'}`}>
            {edgePts > 0 ? '+' : ''}{edgePts.toFixed(1)}
          </span>
        ) : (
          <span className="text-slate-600">-</span>
        )}
        <p className="text-[10px] text-slate-500">Edge</p>
      </div>
    </div>
  );
};

const GameCard = ({ game }) => {
  const predictions = game.predictions || {};
  const dkLines = game.dk_lines || {};
  const edges = game.edges || {};
  
  const spreadPred = predictions.spread?.value;
  const spreadDK = dkLines.spread?.home_line;
  const totalPred = predictions.total?.value;
  const totalDK = dkLines.total?.line;
  
  // Safe checks for edge recommendations
  const spreadEdge = edges.spread || null;
  const totalEdge = edges.total || null;
  const hasSpreadEdge = spreadEdge && spreadEdge.recommendation && spreadEdge.recommendation !== 'no_bet';
  const hasTotalEdge = totalEdge && totalEdge.recommendation && totalEdge.recommendation !== 'no_bet';
  const hasBet = hasSpreadEdge || hasTotalEdge;
  
  return (
    <div className={`card overflow-hidden ${hasBet ? 'border-emerald-500/30' : ''}`}>
      {/* Header */}
      <div className={`px-4 py-3 flex items-center justify-between border-b border-white/5 ${hasBet ? 'bg-emerald-500/10' : 'bg-slate-800/50'}`}>
        <div className="flex items-center gap-3">
          <span className="text-lg font-bold">{game.away_team}</span>
          <span className="text-slate-500">@</span>
          <span className="text-lg font-bold">{game.home_team}</span>
        </div>
        <div className="flex items-center gap-2">
          {hasBet && <span className="badge-bet">BET</span>}
          {game.start_time && (
            <span className="text-xs text-slate-400">
              {new Date(game.start_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          )}
        </div>
      </div>
      
      {/* Predictions vs Lines */}
      <div className="p-4">
        {/* Column Headers */}
        <div className="grid grid-cols-4 gap-2 mb-2 text-[10px] text-slate-500 uppercase">
          <div>Prop</div>
          <div className="text-center">Predicted</div>
          <div className="text-center">Line</div>
          <div className="text-center">Diff</div>
        </div>
        
        {/* Spread */}
        <ComparisonRow 
          label="Spread"
          predicted={spreadPred}
          actual={spreadDK ? parseFloat(spreadDK) : null}
          edge={spreadEdge}
          homeTeam={game.home_team}
        />
        
        {/* Total */}
        <ComparisonRow 
          label="Total"
          predicted={totalPred}
          actual={totalDK ? parseFloat(totalDK) : null}
          edge={totalEdge}
        />
        
        {/* Recommendations */}
        {(hasSpreadEdge || hasTotalEdge) && (
          <div className="mt-4 pt-4 border-t border-white/10">
            <h4 className="text-xs text-slate-400 uppercase mb-2">Recommendations</h4>
            <div className="flex flex-wrap gap-2">
              {hasSpreadEdge && spreadEdge && (
                <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg px-3 py-2">
                  <p className="text-xs text-slate-400">Spread</p>
                  <p className="font-semibold text-emerald-400">
                    {spreadEdge.recommendation === 'home' ? game.home_team : game.away_team}{' '}
                    {spreadEdge.recommendation === 'home' 
                      ? (spreadDK ? (parseFloat(spreadDK) > 0 ? '+' : '') + spreadDK : '')
                      : (dkLines.spread?.away_line ? (parseFloat(dkLines.spread.away_line) > 0 ? '+' : '') + dkLines.spread.away_line : '')
                    }
                  </p>
                  <p className="text-[10px] text-slate-500">
                    {Math.abs(spreadEdge.edge_points || 0).toFixed(1)} pt edge • {spreadEdge.confidence || 'low'} conf
                  </p>
                  {spreadEdge.explanation && (
                    <p className="text-[10px] text-slate-400 mt-1">{spreadEdge.explanation}</p>
                  )}
                </div>
              )}
              {hasTotalEdge && totalEdge && (
                <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-2">
                  <p className="text-xs text-slate-400">Total</p>
                  <p className="font-semibold text-amber-400">
                    {(totalEdge.recommendation || '').toUpperCase()} {totalDK}
                  </p>
                  <p className="text-[10px] text-slate-500">
                    {Math.abs(totalEdge.edge_points || 0).toFixed(1)} pt edge • {totalEdge.confidence || 'low'} conf
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* DK Odds Details */}
        {dkLines.moneyline && (
          <div className="mt-4 pt-4 border-t border-white/10">
            <h4 className="text-xs text-slate-400 uppercase mb-2">DraftKings Odds</h4>
            <div className="grid grid-cols-3 gap-3 text-center text-sm">
              <div className="bg-slate-800/50 rounded-lg py-2">
                <p className="text-[10px] text-slate-500">Moneyline</p>
                <p className="font-mono">
                  <span className="text-slate-400">{game.away_team}</span>{' '}
                  <span className={dkLines.moneyline.away_odds?.startsWith('-') ? 'text-white' : 'text-emerald-400'}>
                    {dkLines.moneyline.away_odds}
                  </span>
                </p>
                <p className="font-mono">
                  <span className="text-slate-400">{game.home_team}</span>{' '}
                  <span className={dkLines.moneyline.home_odds?.startsWith('-') ? 'text-white' : 'text-emerald-400'}>
                    {dkLines.moneyline.home_odds}
                  </span>
                </p>
              </div>
              <div className="bg-slate-800/50 rounded-lg py-2">
                <p className="text-[10px] text-slate-500">Spread</p>
                <p className="font-mono text-slate-300">
                  {dkLines.spread?.home_line > 0 ? '+' : ''}{dkLines.spread?.home_line || '-'}
                </p>
                <p className="text-[10px] text-slate-500">{dkLines.spread?.home_odds}</p>
              </div>
              <div className="bg-slate-800/50 rounded-lg py-2">
                <p className="text-[10px] text-slate-500">Total</p>
                <p className="font-mono text-slate-300">{dkLines.total?.line || '-'}</p>
                <p className="text-[10px] text-slate-500">
                  O {dkLines.total?.over_odds} / U {dkLines.total?.under_odds}
                </p>
              </div>
            </div>
          </div>
        )}
        
        {/* Model Confidence */}
        {(predictions.spread?.confidence || predictions.total?.confidence) && (
          <div className="mt-4 pt-4 border-t border-white/10">
            <h4 className="text-xs text-slate-400 uppercase mb-2">Model Confidence</h4>
            <div className="grid grid-cols-2 gap-3">
              {predictions.spread && (
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-500">Spread</span>
                    <span className="text-slate-300">{((predictions.spread.confidence || 0) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-emerald-500 rounded-full"
                      style={{ width: `${(predictions.spread.confidence || 0) * 100}%` }}
                    />
                  </div>
                  {predictions.spread.lower && predictions.spread.upper && (
                    <p className="text-[10px] text-slate-500 mt-1">
                      Range: {predictions.spread.lower.toFixed(1)} to {predictions.spread.upper.toFixed(1)}
                    </p>
                  )}
                </div>
              )}
              {predictions.total && (
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-500">Total</span>
                    <span className="text-slate-300">{((predictions.total.confidence || 0) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-amber-500 rounded-full"
                      style={{ width: `${(predictions.total.confidence || 0) * 100}%` }}
                    />
                  </div>
                  {predictions.total.lower && predictions.total.upper && (
                    <p className="text-[10px] text-slate-500 mt-1">
                      Range: {predictions.total.lower.toFixed(1)} to {predictions.total.upper.toFixed(1)}
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      
      {/* Footer */}
      <div className="bg-slate-800/30 px-4 py-2 flex justify-between items-center border-t border-white/5">
        <span className="text-xs text-slate-500">Game ID: {game.game_id}</span>
        <Link 
          to={`/nba/game/${game.game_id}`}
          className="text-xs text-emerald-400 hover:text-emerald-300"
        >
          View Player Props →
        </Link>
      </div>
    </div>
  );
};

const NBAGames = () => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const today = new Date().toISOString().split('T')[0];
      const res = selectedDate === today
        ? await api.getNBAGamePredictionsToday()
        : await api.getNBAGamePredictionsByDate(selectedDate);
      setData(res);
    } catch (e) {
      console.error('Failed to load game predictions:', e);
      setData({ games: [], total_games: 0 });
    }
    setLoading(false);
  }, [selectedDate]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const games = data?.games || [];
  const gamesWithBets = games.filter(g => {
    const sEdge = g.edges?.spread;
    const tEdge = g.edges?.total;
    return (sEdge && sEdge.recommendation && sEdge.recommendation !== 'no_bet') || 
           (tEdge && tEdge.recommendation && tEdge.recommendation !== 'no_bet');
  });

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
            <span>🏀</span> NBA Game Predictions
            {data?.odds_available && <span className="badge bg-emerald-500/20 text-emerald-400 text-xs">LIVE ODDS</span>}
          </h1>
          <p className="text-slate-400 text-sm mt-0.5">
            {games.length} games • {gamesWithBets.length} with betting edge
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

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Total Games</p>
          <p className="text-xl font-bold">{games.length}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Spread Bets</p>
          <p className="text-xl font-bold text-emerald-400">{data?.spread_recommendations || 0}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Total Bets</p>
          <p className="text-xl font-bold text-amber-400">{data?.total_recommendations || 0}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">DK Odds</p>
          <p className="text-xl font-bold">{data?.odds_available ? '✓' : '✗'}</p>
        </div>
      </div>

      {/* Quick Navigation */}
      <div className="flex gap-2">
        <Link to="/nba" className="btn-secondary text-sm">
          👤 Player Props
        </Link>
        <Link to="/nba/bets" className="btn-secondary text-sm">
          💰 Bet History
        </Link>
      </div>

      {/* Games with Bets First */}
      {gamesWithBets.length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-emerald-400 uppercase tracking-wide mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            Recommended Bets ({gamesWithBets.length})
          </h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {gamesWithBets.map((game, idx) => (
              <GameCard key={game.game_id || idx} game={game} />
            ))}
          </div>
        </section>
      )}

      {/* Other Games */}
      {games.filter(g => !gamesWithBets.includes(g)).length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">
            Other Games ({games.length - gamesWithBets.length})
          </h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {games.filter(g => !gamesWithBets.includes(g)).map((game, idx) => (
              <GameCard key={game.game_id || idx} game={game} />
            ))}
          </div>
        </section>
      )}

      {games.length === 0 && (
        <div className="card p-12 text-center">
          <span className="text-4xl block mb-2">🏀</span>
          <p className="text-slate-400">No game predictions found for this date</p>
          <p className="text-slate-500 text-sm mt-1">
            Make sure predictions have been generated and odds have been scraped
          </p>
        </div>
      )}

      {/* Legend */}
      <div className="card p-4">
        <h3 className="font-semibold mb-3">Understanding the Predictions</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-slate-400">
          <div>
            <h4 className="text-emerald-400 font-medium mb-2">Spread Betting</h4>
            <p className="mb-2">Spreads are shown in DraftKings convention:</p>
            <ul className="space-y-1 text-xs">
              <li>• <span className="text-white">-8.5</span> = Favored to win by 8.5+</li>
              <li>• <span className="text-white">+8.5</span> = Underdog, can lose by up to 8</li>
            </ul>
            <p className="mt-2 text-xs">Positive edge on HOME = our model thinks home covers better than the line suggests.</p>
          </div>
          <div>
            <h4 className="text-amber-400 font-medium mb-2">Total (Over/Under)</h4>
            <p className="mb-2">Combined points scored by both teams.</p>
            <ul className="space-y-1 text-xs">
              <li>• <span className="text-emerald-400">OVER</span> = Model predicts more points than the line</li>
              <li>• <span className="text-red-400">UNDER</span> = Model predicts fewer points than the line</li>
            </ul>
          </div>
        </div>
        <div className="mt-4 pt-4 border-t border-white/10">
          <h4 className="text-white font-medium mb-2">Edge Thresholds</h4>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div>
              <span className="text-emerald-400">High Confidence</span>
              <p className="text-slate-500">Spread: 3+ pts | Total: 5+ pts</p>
            </div>
            <div>
              <span className="text-amber-400">Medium Confidence</span>
              <p className="text-slate-500">Spread: 1.5-3 pts | Total: 3-5 pts</p>
            </div>
            <div>
              <span className="text-slate-400">No Bet</span>
              <p className="text-slate-500">Edge below threshold</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NBAGames;