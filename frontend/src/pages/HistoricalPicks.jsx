// src/pages/HistoricalPicks.jsx
import React, { useState, useEffect } from 'react';
import api from '../services/api';

const HistoricalPicks = () => {
  const [loading, setLoading] = useState(true);
  const [predictions, setPredictions] = useState([]);
  const [days, setDays] = useState(7);
  const [filterProp, setFilterProp] = useState('all');
  const [filterResult, setFilterResult] = useState('all');
  const [page, setPage] = useState(1);
  const perPage = 50;

  useEffect(() => {
    loadHistory();
  }, [days]);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const res = await api.getNBAHistory(days);
      setPredictions(res.predictions || []);
      setPage(1);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Get unique prop types
  const propTypes = [...new Set(predictions.map(p => p.prop_type))].sort();

  // Filter predictions
  let filtered = predictions.filter(p => {
    if (filterProp !== 'all' && p.prop_type !== filterProp) return false;
    // filterResult would need actual results data to work
    return true;
  });

  // Sort by date (newest first)
  filtered.sort((a, b) => b.prediction_date.localeCompare(a.prediction_date));

  // Paginate
  const totalPages = Math.ceil(filtered.length / perPage);
  const paginated = filtered.slice((page - 1) * perPage, page * perPage);

  // Group by date
  const groupedByDate = paginated.reduce((acc, pred) => {
    const date = pred.prediction_date;
    if (!acc[date]) acc[date] = [];
    acc[date].push(pred);
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            📜 Historical Picks
          </h1>
          <p className="text-gray-400">
            {predictions.length.toLocaleString()} predictions from the last {days} days
          </p>
        </div>
        
        <div className="flex items-center gap-4">
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
          >
            <option value={7}>Last 7 Days</option>
            <option value={14}>Last 14 Days</option>
            <option value={30}>Last 30 Days</option>
            <option value={60}>Last 60 Days</option>
          </select>
          <button
            onClick={loadHistory}
            className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700 flex flex-wrap gap-4 items-center">
        <div>
          <label className="text-sm text-gray-400 block mb-1">Prop Type</label>
          <select
            value={filterProp}
            onChange={(e) => { setFilterProp(e.target.value); setPage(1); }}
            className="bg-gray-700 border border-gray-600 rounded px-3 py-2"
          >
            <option value="all">All Props</option>
            {propTypes.map(p => (
              <option key={p} value={p}>{p.toUpperCase()}</option>
            ))}
          </select>
        </div>
        
        <div className="ml-auto text-sm text-gray-400">
          Showing {paginated.length} of {filtered.length} picks
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-green-500"></div>
        </div>
      )}

      {/* Grouped Predictions */}
      {!loading && Object.keys(groupedByDate).length > 0 && (
        <div className="space-y-6">
          {Object.entries(groupedByDate).map(([date, preds]) => (
            <div key={date} className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
              <div className="bg-gray-700 px-4 py-3 flex justify-between items-center">
                <h3 className="font-semibold">{new Date(date).toLocaleDateString('en-US', { 
                  weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' 
                })}</h3>
                <span className="text-sm text-gray-400">{preds.length} predictions</span>
              </div>
              
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-750 text-sm text-gray-400">
                    <tr>
                      <th className="py-2 px-4 text-left">Player</th>
                      <th className="py-2 px-4 text-left">Team</th>
                      <th className="py-2 px-4 text-left">Prop</th>
                      <th className="py-2 px-4 text-right">Prediction</th>
                      <th className="py-2 px-4 text-right">Line</th>
                      <th className="py-2 px-4 text-right">Confidence</th>
                      <th className="py-2 px-4 text-center">Call</th>
                    </tr>
                  </thead>
                  <tbody>
                    {preds.map((pred, idx) => (
                      <tr key={idx} className="border-t border-gray-700 hover:bg-gray-750">
                        <td className="py-2 px-4">{pred.player_name}</td>
                        <td className="py-2 px-4">{pred.team_abbrev} vs {pred.opponent_abbrev}</td>
                        <td className="py-2 px-4 uppercase font-semibold">{pred.prop_type}</td>
                        <td className="py-2 px-4 text-right font-mono">
                          {parseFloat(pred.predicted_value).toFixed(1)}
                        </td>
                        <td className="py-2 px-4 text-right font-mono">
                          {pred.line || '-'}
                        </td>
                        <td className="py-2 px-4 text-right">
                          <span className={`${
                            pred.confidence >= 0.7 ? 'text-green-400' : 
                            pred.confidence >= 0.6 ? 'text-yellow-400' : 'text-gray-400'
                          }`}>
                            {(pred.confidence * 100).toFixed(0)}%
                          </span>
                        </td>
                        <td className="py-2 px-4 text-center">
                          {pred.recommended_bet && (
                            <span className={`px-2 py-1 rounded text-xs font-bold ${
                              pred.recommended_bet === 'over' ? 'bg-green-600' : 'bg-red-600'
                            }`}>
                              {pred.recommended_bet.toUpperCase()}
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* No Data */}
      {!loading && Object.keys(groupedByDate).length === 0 && (
        <div className="bg-gray-800 rounded-xl p-12 border border-gray-700 text-center text-gray-500">
          No predictions found for the selected filters
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-center items-center gap-4">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            className="px-4 py-2 bg-gray-700 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-600"
          >
            Previous
          </button>
          <span className="text-gray-400">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="px-4 py-2 bg-gray-700 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-600"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
};

export default HistoricalPicks;