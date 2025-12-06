// src/services/api.js
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

  async getNBAParlays(date = null) {
    const dateParam = date ? `?date=${date}` : '';
    return this.fetch(`/nba/predictions/parlays${dateParam}`);
  }

  async getNBAGameDetail(gameId) {
    return this.fetch(`/nba/games/${gameId}`);
  }

  async getNBAAccuracy(days = 30) {
    return this.fetch(`/nba/accuracy?days=${days}`);
  }

  // NHL Endpoints
  async getNHLTodayPredictions() {
    return this.fetch('/nhl/predictions/today');
  }

  async getNHLHistory(days = 30) {
    return this.fetch(`/nhl/predictions/history?days=${days}`);
  }

  async getNHLSummary(days = 30) {
    return this.fetch(`/nhl/predictions/summary?days=${days}`);
  }
}

export const api = new ApiService();
export default api;