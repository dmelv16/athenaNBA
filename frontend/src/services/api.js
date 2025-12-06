// src/services/api.js - Enhanced API Service
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

  // NBA Endpoints
  async getNBATodayPredictions() {
    return this.fetch('/nba/predictions/today');
  }

  async getNBAPredictionsByDate(date) {
    return this.fetch(`/nba/predictions/date/${date}`);
  }

  async getNBAHistory(days = 30) {
    return this.fetch(`/nba/predictions/history?days=${days}`);
  }

  async getNBABestBets(minConfidence = 0.65, minEdge = 2.0) {
    return this.fetch(`/nba/predictions/best-bets?min_confidence=${minConfidence}&min_edge=${minEdge}`);
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

  // NHL Endpoints
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

  // Get current bankroll status
  async getBankrollStatus() {
    return this.fetch('/bankroll/status');
  }

  // Update bankroll
  async updateBankroll(amount) {
    return this.fetch('/bankroll/update', {
      method: 'POST',
      body: JSON.stringify({ bankroll: amount })
    });
  }
}

export const api = new ApiService();
export default api;