// src/pages/NHLPredictions.jsx
import React, { useState, useEffect, useCallback } from 'react';
import api from '../services/api';

const NHLPredictions = () => {
  const [loading, setLoading] = useState(true);
  const [todayData, setTodayData] = useState(null);
  const [historyData, setHistoryData] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [summary, setSummary] = useState(null);
  const [activeTab, setActiveTab] = useState('today');
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [today, hist, acc, sum] = await Promise.allSettled([
        api.getNHLTodayPredictions(),
        api.getNHLHistory(30),
        api.getNHLAccuracy(30),
        api.getNHLSummary(30),
      ]);
      if (today.status === 'fulfilled') setTodayData(today.value);
      if (hist.status === 'fulfilled') setHistoryData(hist.value);
      if (acc.status === 'fulfilled') setAccuracy(acc.value);
      if (sum.status === 'fulfilled') setSummary(sum.value);
    } catch (e) { console.error(e); }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const loadByDate = async (date) => {
    setLoading(true);
    try {
      const res = await api.getNHLPredictionsByDate(date);
      setTodayData(res);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  // Helper to determine if a game should be bet on
  const isBetRecommended = (game) => {
    if (game.action === 'BET') return true;
    // Use the actual bet_pct_bankroll from database
    const betPct = parseFloat(game.bet_pct_bankroll || 0);
    if (betPct > 0) return true;
    const betSize = parseFloat(game.bet_size || 0);
    if (betSize > 0) return true;
    return false;
  };

  // Get bet percentage directly from database field
  const getBetPct = (game) => {
    // bet_pct_bankroll is stored as a decimal (e.g., 0.025 = 2.5%)
    const betPct = parseFloat(game.bet_pct_bankroll || 0);
    if (betPct > 0) {
      // If it's already a percentage (> 1), use as is, otherwise multiply by 100
      return betPct > 1 ? betPct.toFixed(1) : (betPct * 100).toFixed(1);
    }
    return '0.0';
  };

  // Get suggested bet amount based on bankroll
  const getBetAmount = (game, bankroll) => {
    const betSize = parseFloat(game.bet_size || 0);
    if (betSize > 0) return betSize.toFixed(2);
    
    const betPct = parseFloat(game.bet_pct_bankroll || 0);
    if (betPct > 0 && bankroll > 0) {
      const pct = betPct > 1 ? betPct / 100 : betPct;
      return (bankroll * pct).toFixed(2);
    }
    return '0.00';
  };

  const games = todayData?.games || todayData?.predictions || [];
  const betGames = games.filter(isBetRecommended);
  const skipGames = games.filter(g => !isBetRecommended(g));

  // Calculate current bankroll from summary or use default
  // In production, this should come from your database/API
  const currentBankroll = summary?.current_bankroll || summary?.bankroll || 1412.77;
  
  // Calculate total stake for today's bets
  const totalStakeToday = betGames.reduce((sum, g) => {
    return sum + parseFloat(getBetAmount(g, currentBankroll));
  }, 0);

  // Calculate total expected value
  const totalEV = betGames.reduce((sum, g) => {
    return sum + parseFloat(g.expected_value || 0);
  }, 0);

  if (loading && !todayData) {
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
            <span>🏒</span> NHL Predictions
          </h1>
          <p className="text-slate-400 text-sm mt-0.5">{games.length} games • {betGames.length} recommended bets</p>
        </div>
        <div className="flex gap-2">
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => { setSelectedDate(e.target.value); loadByDate(e.target.value); }}
            className="input-field"
          />
          <button onClick={loadData} className="btn-primary text-sm">Refresh</button>
        </div>
      </div>

      {/* Bankroll & Today's Summary */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="card p-3 border-emerald-500/20 bg-emerald-500/5">
          <p className="text-[10px] text-slate-500 uppercase">Current Bankroll</p>
          <p className="text-xl font-bold text-emerald-400">${currentBankroll.toFixed(2)}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Today's Bets</p>
          <p className="text-xl font-bold">{betGames.length}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Total Stake</p>
          <p className="text-xl font-bold text-amber-400">${totalStakeToday.toFixed(2)}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Expected Value</p>
          <p className={`text-xl font-bold ${totalEV >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {totalEV >= 0 ? '+' : ''}${totalEV.toFixed(2)}
          </p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">30-Day Accuracy</p>
          <p className={`text-xl font-bold ${(accuracy?.overall_accuracy || 0) > 0.52 ? 'text-emerald-400' : 'text-red-400'}`}>
            {((accuracy?.overall_accuracy || 0) * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-white/5">
        {['today', 'history', 'performance'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium transition-colors capitalize ${
              activeTab === tab 
                ? 'text-emerald-400 border-b-2 border-emerald-400' 
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {tab === 'today' ? "Today's Games" : tab}
          </button>
        ))}
      </div>

      {/* Today Tab */}
      {activeTab === 'today' && (
        <div className="space-y-6">
          {/* Recommended Bets Section */}
          {betGames.length > 0 && (
            <section>
              <h3 className="text-sm font-semibold text-emerald-400 uppercase tracking-wide mb-3 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                Recommended Bets ({betGames.length})
              </h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {betGames.map((game, idx) => {
                  const betPct = getBetPct(game);
                  const betAmount = getBetAmount(game, currentBankroll);
                  const predictedTeam = game.predicted_winner === 'HOME' ? game.home_team : game.away_team;
                  
                  return (
                    <div key={game.game_id || idx} className="card border-emerald-500/30 bg-emerald-500/5 overflow-hidden">
                      <div className="bg-emerald-500/10 px-4 py-2 flex items-center justify-between">
                        <span className="badge-bet">RECOMMENDED BET</span>
                        <span className="text-xs text-slate-400">
                          {game.game_time ? new Date(game.game_time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : ''}
                        </span>
                      </div>
                      <div className="p-4">
                        <div className="flex items-center justify-between mb-4">
                          <div className="text-center flex-1">
                            <p className="text-xl font-bold">{game.away_team}</p>
                            <p className="text-[10px] text-slate-500">AWAY</p>
                          </div>
                          <span className="text-slate-600 text-xl px-4">@</span>
                          <div className="text-center flex-1">
                            <p className="text-xl font-bold">{game.home_team}</p>
                            <p className="text-[10px] text-slate-500">HOME</p>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-4 gap-2 text-center mb-4">
                          <div className="bg-slate-800/50 rounded-lg py-2">
                            <p className="text-[10px] text-slate-500 mb-0.5">Pick</p>
                            <p className="font-bold text-emerald-400 text-sm">{predictedTeam}</p>
                          </div>
                          <div className="bg-slate-800/50 rounded-lg py-2">
                            <p className="text-[10px] text-slate-500 mb-0.5">Prob</p>
                            <p className="font-bold text-sm">{((game.model_probability || 0) * 100).toFixed(0)}%</p>
                          </div>
                          <div className="bg-slate-800/50 rounded-lg py-2">
                            <p className="text-[10px] text-slate-500 mb-0.5">Edge</p>
                            <p className="font-bold text-emerald-400 text-sm">+{((game.edge || 0) * 100).toFixed(1)}%</p>
                          </div>
                          <div className="bg-slate-800/50 rounded-lg py-2">
                            <p className="text-[10px] text-slate-500 mb-0.5">Odds</p>
                            <p className="font-bold text-sm">{game.american_odds > 0 ? '+' : ''}{game.american_odds || '-'}</p>
                          </div>
                        </div>
                        
                        <div className="bg-emerald-500/10 rounded-lg p-3">
                          <div className="flex justify-between items-center mb-2">
                            <span className="text-sm text-slate-300">Bankroll %</span>
                            <span className="font-bold text-emerald-400">{betPct}%</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-slate-300">Suggested Bet</span>
                            <span className="font-bold text-emerald-400 text-lg">${betAmount}</span>
                          </div>
                          {game.expected_value && parseFloat(game.expected_value) !== 0 && (
                            <div className="flex justify-between items-center mt-2 pt-2 border-t border-white/10">
                              <span className="text-xs text-slate-400">Expected Value</span>
                              <span className={`text-sm font-semibold ${parseFloat(game.expected_value) > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                {parseFloat(game.expected_value) > 0 ? '+' : ''}${parseFloat(game.expected_value).toFixed(2)}
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* Skip Section */}
          {skipGames.length > 0 && (
            <section>
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-slate-500"></span>
                No Bet Recommended ({skipGames.length})
              </h3>
              <div className="card overflow-hidden">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Matchup</th>
                      <th>Pick</th>
                      <th className="text-right">Probability</th>
                      <th className="text-right">Edge</th>
                      <th className="text-center">Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {skipGames.map((game, idx) => (
                      <tr key={game.game_id || idx}>
                        <td className="font-medium">{game.away_team} @ {game.home_team}</td>
                        <td className="text-slate-300">
                          {game.predicted_winner === 'HOME' ? game.home_team : game.away_team}
                        </td>
                        <td className="text-right">{((game.model_probability || 0) * 100).toFixed(0)}%</td>
                        <td className={`text-right ${(game.edge || 0) > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {(game.edge || 0) > 0 ? '+' : ''}{((game.edge || 0) * 100).toFixed(1)}%
                        </td>
                        <td className="text-center">
                          <span className="badge-skip">
                            {(game.edge || 0) <= 0 ? 'Negative Edge' : 'Insufficient Edge'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {games.length === 0 && (
            <div className="card p-12 text-center">
              <span className="text-4xl block mb-2">🏒</span>
              <p className="text-slate-400">No games scheduled for this date</p>
            </div>
          )}
        </div>
      )}

      {/* History Tab */}
      {activeTab === 'history' && historyData && (
        <div className="card overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Matchup</th>
                <th>Pick</th>
                <th className="text-right">Prob</th>
                <th className="text-right">Edge</th>
                <th className="text-center">Action</th>
                <th className="text-center">Result</th>
              </tr>
            </thead>
            <tbody>
              {(historyData.predictions || []).slice(0, 50).map((pred, idx) => (
                <tr key={idx}>
                  <td className="text-slate-400 text-sm">{pred.prediction_date}</td>
                  <td className="font-medium">{pred.away_team} @ {pred.home_team}</td>
                  <td className="text-emerald-400">
                    {pred.predicted_winner === 'HOME' ? pred.home_team : pred.away_team}
                  </td>
                  <td className="text-right">{((pred.model_probability || 0) * 100).toFixed(0)}%</td>
                  <td className={`text-right ${(pred.edge || 0) > 0 ? 'text-emerald-400' : 'text-slate-400'}`}>
                    {(pred.edge || 0) > 0 ? '+' : ''}{((pred.edge || 0) * 100).toFixed(1)}%
                  </td>
                  <td className="text-center">
                    <span className={`badge ${pred.action === 'BET' ? 'badge-bet' : 'badge-skip'}`}>
                      {pred.action || 'SKIP'}
                    </span>
                  </td>
                  <td className="text-center">
                    {pred.was_correct !== null && pred.was_correct !== undefined ? (
                      <span className={pred.was_correct ? 'badge-win' : 'badge-loss'}>
                        {pred.was_correct ? 'WIN' : 'LOSS'}
                      </span>
                    ) : (
                      <span className="badge-pending">PENDING</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Performance Tab */}
      {activeTab === 'performance' && accuracy && (
        <div className="space-y-4">
          {accuracy.by_confidence && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {accuracy.by_confidence.map((bucket, idx) => (
                <div key={idx} className="card p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-slate-300 capitalize">{bucket.bucket} Confidence</span>
                    <span className="text-xs text-slate-500">{bucket.count} games</span>
                  </div>
                  <p className={`text-3xl font-bold ${bucket.accuracy > 0.55 ? 'text-emerald-400' : bucket.accuracy > 0.50 ? 'text-amber-400' : 'text-red-400'}`}>
                    {(bucket.accuracy * 100).toFixed(1)}%
                  </p>
                  <div className="mt-3 h-2 bg-slate-700 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full ${bucket.accuracy > 0.55 ? 'bg-emerald-500' : bucket.accuracy > 0.50 ? 'bg-amber-500' : 'bg-red-500'}`}
                      style={{width: `${bucket.accuracy * 100}%`}}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
          
          <div className="card p-4">
            <h3 className="font-semibold mb-4">Understanding the Numbers</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-slate-300">
              <div>
                <h4 className="font-semibold text-emerald-400 mb-2">Profitability Thresholds</h4>
                <ul className="space-y-1 text-slate-400">
                  <li>• 52.4% accuracy = breakeven at -110 odds</li>
                  <li>• 55% accuracy ≈ 4.5% ROI</li>
                  <li>• 60% accuracy ≈ 14% ROI</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-amber-400 mb-2">Key Metrics</h4>
                <ul className="space-y-1 text-slate-400">
                  <li>• <strong>Edge:</strong> Model probability minus implied odds</li>
                  <li>• <strong>Bet %:</strong> Kelly-optimized stake from database</li>
                  <li>• <strong>EV:</strong> Expected value of the bet</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NHLPredictions;