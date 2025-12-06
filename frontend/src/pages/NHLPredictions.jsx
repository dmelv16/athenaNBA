// src/pages/NHLPredictions.jsx
import React, { useState, useEffect, useCallback } from 'react';
import api from '../services/api';

const NHLPredictions = () => {
  const [loading, setLoading] = useState(true);
  const [todayData, setTodayData] = useState(null);
  const [historyData, setHistoryData] = useState(null);
  const [betResults, setBetResults] = useState(null); // NEW: actual bet results
  const [accuracy, setAccuracy] = useState(null);
  const [trackingStats, setTrackingStats] = useState(null);
  const [activeTab, setActiveTab] = useState('today');
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [today, hist, acc, stats, bets] = await Promise.allSettled([
        api.getNHLTodayPredictions(),
        api.getNHLHistory(30),
        api.getNHLAccuracy(30),
        api.getTrackingStats(),
        api.getBetResults(30, 100), // Get actual bet results
      ]);
      if (today.status === 'fulfilled') setTodayData(today.value);
      if (hist.status === 'fulfilled') setHistoryData(hist.value);
      if (acc.status === 'fulfilled') setAccuracy(acc.value);
      if (stats.status === 'fulfilled') setTrackingStats(stats.value);
      if (bets.status === 'fulfilled') setBetResults(bets.value);
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

  const isBetRecommended = (game) => {
    if (game.action === 'BET') return true;
    const betPct = parseFloat(game.bet_pct_bankroll || 0);
    if (betPct > 0) return true;
    const betSize = parseFloat(game.bet_size || 0);
    if (betSize > 0) return true;
    return false;
  };

  const getBetPct = (game) => {
    const betPct = parseFloat(game.bet_pct_bankroll || 0);
    if (betPct > 0) {
      return betPct > 1 ? betPct.toFixed(1) : (betPct * 100).toFixed(1);
    }
    return '0.0';
  };

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

  const overall = trackingStats?.overall || {};
  const currentBankroll = trackingStats?.current_bankroll || 1000;
  
  const totalStakeToday = betGames.reduce((sum, g) => {
    return sum + parseFloat(getBetAmount(g, currentBankroll));
  }, 0);

  // Use bet_results for history (these have actual results)
  const historyBets = betResults?.bets || [];

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
      <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
        <div className="card p-3 border-emerald-500/20 bg-emerald-500/5">
          <p className="text-[10px] text-slate-500 uppercase">Bankroll</p>
          <p className="text-xl font-bold text-emerald-400">${currentBankroll.toFixed(2)}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Record</p>
          <p className="text-xl font-bold">{overall.wins || 0}W - {overall.losses || 0}L</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Total P&L</p>
          <p className={`text-xl font-bold ${(overall.total_pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {(overall.total_pnl || 0) >= 0 ? '+' : ''}${(overall.total_pnl || 0).toFixed(0)}
          </p>
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
          <p className="text-[10px] text-slate-500 uppercase">Win Rate</p>
          <p className={`stat-lg ${(overall.win_rate || 0) > 0.52 ? 'text-emerald-400' : 'text-amber-400'}`}>
            {((overall.win_rate || 0) * 100).toFixed(1)}%
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
            {tab === 'today' ? "Today's Games" : tab === 'history' ? 'Bet History' : tab}
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

      {/* History Tab - Now shows only BETS with actual results */}
      {activeTab === 'history' && (
        <div className="card overflow-hidden">
          {historyBets.length > 0 ? (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Matchup</th>
                  <th>Pick</th>
                  <th className="text-center">Score</th>
                  <th className="text-right">Odds</th>
                  <th className="text-right">Bet</th>
                  <th className="text-center">Result</th>
                  <th className="text-right">P&L</th>
                </tr>
              </thead>
              <tbody>
                {historyBets.map((bet, idx) => (
                  <tr key={bet.game_id || idx}>
                    <td className="text-slate-400 text-sm">{bet.game_date}</td>
                    <td className="font-medium">{bet.away_team} @ {bet.home_team}</td>
                    <td className="text-emerald-400">{bet.predicted_team}</td>
                    <td className="text-center text-slate-300">
                      {bet.away_score !== null && bet.home_score !== null 
                        ? `${bet.away_score} - ${bet.home_score}`
                        : '-'
                      }
                    </td>
                    <td className="text-right">
                      {bet.american_odds > 0 ? '+' : ''}{bet.american_odds || '-'}
                    </td>
                    <td className="text-right">${parseFloat(bet.bet_size || 0).toFixed(2)}</td>
                    <td className="text-center">
                      <span className={bet.bet_result === 'WIN' ? 'badge-win' : 'badge-loss'}>
                        {bet.bet_result}
                      </span>
                    </td>
                    <td className={`text-right font-semibold ${parseFloat(bet.pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {parseFloat(bet.pnl || 0) >= 0 ? '+' : ''}${parseFloat(bet.pnl || 0).toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="p-12 text-center">
              <span className="text-4xl block mb-2">📊</span>
              <p className="text-slate-400">No bet history available yet</p>
            </div>
          )}
        </div>
      )}

      {/* Performance Tab */}
      {activeTab === 'performance' && (
        <div className="space-y-4">
          {/* Stats Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="card p-4">
              <p className="stat-label">Total Bets</p>
              <p className="stat-lg">{overall.total_bets || 0}</p>
            </div>
            <div className="card p-4">
              <p className="stat-label">Win Rate</p>
              <p className={`stat-lg ${(overall.win_rate || 0) > 0.52 ? 'text-emerald-400' : 'text-red-400'}`}>
                {((overall.win_rate || 0) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="card p-4">
              <p className="stat-label">Total P&L</p>
              <p className={`stat-lg ${(overall.total_pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {(overall.total_pnl || 0) >= 0 ? '+' : ''}${(overall.total_pnl || 0).toFixed(2)}
              </p>
            </div>
            <div className="card p-4">
              <p className="stat-label">ROI</p>
              <p className={`stat-lg ${(overall.roi_pct || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {(overall.roi_pct || 0) >= 0 ? '+' : ''}{(overall.roi_pct || 0).toFixed(1)}%
              </p>
            </div>
          </div>

          {/* By Edge Class */}
          {trackingStats?.by_edge_class && trackingStats.by_edge_class.length > 0 && (
            <div className="card p-4">
              <h3 className="font-semibold mb-4">Performance by Edge Class</h3>
              <div className="space-y-3">
                {trackingStats.by_edge_class.map((edge, idx) => (
                  <div key={idx} className="flex items-center justify-between p-3 bg-slate-800/30 rounded-lg">
                    <div>
                      <p className="font-semibold">{edge.edge_class}</p>
                      <p className="text-xs text-slate-500">{edge.bets} bets • {edge.wins}W • Avg Edge: {(edge.avg_edge * 100).toFixed(1)}%</p>
                    </div>
                    <div className="text-right">
                      <p className={`text-xl font-bold ${edge.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {edge.pnl >= 0 ? '+' : ''}${edge.pnl.toFixed(2)}
                      </p>
                      <p className="text-xs text-slate-500">{((edge.win_rate || 0) * 100).toFixed(0)}% win rate</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Top Teams */}
          {trackingStats?.top_teams && trackingStats.top_teams.length > 0 && (
            <div className="card p-4">
              <h3 className="font-semibold mb-4">Top Performing Teams</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {trackingStats.top_teams.map((team, i) => (
                  <div key={i} className="flex items-center justify-between p-3 bg-slate-800/30 rounded-lg">
                    <div className="flex items-center gap-3">
                      <span className="text-lg font-bold text-slate-500">#{i + 1}</span>
                      <div>
                        <p className="font-medium">{team.team}</p>
                        <p className="text-xs text-slate-500">{team.bets} bets • {team.wins}W</p>
                      </div>
                    </div>
                    <p className={`font-bold ${team.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {team.pnl >= 0 ? '+' : ''}${team.pnl.toFixed(2)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Accuracy by Confidence */}
          {accuracy?.by_confidence && (
            <div className="card p-4">
              <h3 className="font-semibold mb-4">Accuracy by Confidence Level</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {accuracy.by_confidence.map((bucket, idx) => (
                  <div key={idx} className="bg-slate-800/50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-slate-300 capitalize">{bucket.bucket}</span>
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
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default NHLPredictions;