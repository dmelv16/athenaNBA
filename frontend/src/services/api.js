// src/services/api.js - Enhanced with Odds Integration
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5001/api';

class ApiService {
  async fetch(endpoint, options = {}) {
    try {
      const response = await fetch(`${API_BASE}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`API Error (${endpoint}):`, error);
      throw error;
    }
  }

  // Health & General
  async getHealth() {
    return this.fetch('/health');
  }

  async getSports() {
    return this.fetch('/sports');
  }

  // ============================================
  // NBA Endpoints - Enhanced with Odds
  // ============================================
  
  async getNBATodayPredictions() {
    return this.fetch('/nba/predictions/today');
  }

  async getNBAPredictionsByDate(date) {
    return this.fetch(`/nba/predictions/date/${date}`);
  }

  async getNBAHistory(days = 30) {
    return this.fetch(`/nba/predictions/history?days=${days}`);
  }

  async getNBAGameDetail(gameId) {
    return this.fetch(`/nba/games/${gameId}`);
  }

  async getNBAAccuracy(days = 30) {
    return this.fetch(`/nba/accuracy?days=${days}`);
  }

  async getNBAPlayerHistory(playerId, limit = 5, propType = null) {
    let url = `/nba/players/${playerId}/history?limit=${limit}`;
    if (propType) url += `&prop_type=${propType}`;
    return this.fetch(url);
  }

  // NEW: Odds-related endpoints
  async getNBAOddsToday() {
    return this.fetch('/nba/odds/today');
  }

  async getNBAOddsForPlayer(playerName, propType = null, date = null) {
    let url = `/nba/odds/player/${encodeURIComponent(playerName)}`;
    const params = [];
    if (propType) params.push(`prop_type=${propType}`);
    if (date) params.push(`date=${date}`);
    if (params.length) url += `?${params.join('&')}`;
    return this.fetch(url);
  }

  async getNBAPredictionsWithEdge(minEdge = 0.02, date = null) {
    let url = `/nba/predictions/with-edge?min_edge=${minEdge}`;
    if (date) url += `&date=${date}`;
    return this.fetch(url);
  }

  // NEW: NBA Bet tracking endpoints
  async saveNBABet(betData) {
    return this.fetch('/nba/bets/save', {
      method: 'POST',
      body: JSON.stringify(betData)
    });
  }

  async saveAllNBABets(options = {}) {
    return this.fetch('/nba/bets/save-all', {
      method: 'POST',
      body: JSON.stringify(options)
    });
  }

  async getNBAPendingBets(date = null) {
    let url = '/nba/bets/pending';
    if (date) url += `?date=${date}`;
    return this.fetch(url);
  }

  async getNBABetHistory(days = 30, result = null) {
    let url = `/nba/bets/history?days=${days}`;
    if (result) url += `&result=${result}`;
    return this.fetch(url);
  }

  async updateNBABetResult(betId, actualValue) {
    return this.fetch('/nba/bets/result', {
      method: 'POST',
      body: JSON.stringify({ bet_id: betId, actual_value: actualValue })
    });
  }

  async getNBABetPerformance(days = 30) {
    return this.fetch(`/nba/bets/performance?days=${days}`);
  }

  // NEW: Game Lines endpoints
  async getNBAGameLinesToday() {
    return this.fetch('/nba/game-lines/today');
  }

  async getNBAGameLinesByDate(date) {
    return this.fetch(`/nba/game-lines/date/${date}`);
  }

  async getNBAGameOdds(gameId) {
    return this.fetch(`/nba/games/${gameId}/odds`);
  }

  // ============================================
  // NHL Endpoints
  // ============================================
  
  async getNHLTodayPredictions() {
    return this.fetch('/nhl/predictions/today');
  }

  async getNHLPredictionsByDate(date) {
    return this.fetch(`/nhl/predictions/date/${date}`);
  }

  async getNHLHistory(days = 30) {
    return this.fetch(`/nhl/predictions/history?days=${days}`);
  }

  async getNHLSummary(days = 30) {
    return this.fetch(`/nhl/predictions/summary?days=${days}`);
  }

  async getNHLBestBets(minEdge = 0.05, minProbability = 0.55) {
    return this.fetch(`/nhl/predictions/best-bets?min_edge=${minEdge}&min_probability=${minProbability}`);
  }

  async getNHLAccuracy(days = 30) {
    return this.fetch(`/nhl/accuracy?days=${days}`);
  }

  async getNHLResults(days = 30) {
    return this.fetch(`/nhl/results/history?days=${days}`);
  }

  // ============================================
  // Bankroll Endpoints
  // ============================================
  
  async getBankrollStatus() {
    return this.fetch('/bankroll/status');
  }

  async updateBankroll(amount, sport = 'nhl') {
    return this.fetch('/bankroll/update', {
      method: 'POST',
      body: JSON.stringify({ bankroll: amount, sport })
    });
  }

  // ============================================
  // NHL Historical Tracking Endpoints
  // ============================================
  
  async getTrackingPerformance(days = 30) {
    return this.fetch(`/tracking/performance?days=${days}`);
  }

  async getBankrollHistory(days = 30) {
    return this.fetch(`/tracking/bankroll?days=${days}`);
  }

  async getBetResults(days = 30, limit = 100, result = null) {
    let url = `/tracking/bets?days=${days}&limit=${limit}`;
    if (result) url += `&result=${result}`;
    return this.fetch(url);
  }

  async getTrackingStats() {
    return this.fetch('/tracking/stats');
  }

  async runTrackingUpdate() {
    return this.fetch('/tracking/update', { method: 'POST' });
  }

  async runTrackingBackfill(days = 30, resetBankroll = false, startingBankroll = 1000) {
    return this.fetch('/tracking/backfill', {
      method: 'POST',
      body: JSON.stringify({
        days,
        reset_bankroll: resetBankroll,
        starting_bankroll: startingBankroll
      })
    });
  }
}

export const api = new ApiService();
export default api;