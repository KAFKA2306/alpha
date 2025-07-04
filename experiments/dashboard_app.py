#!/usr/bin/env python3
"""
Real-time Dashboard Application for MCP Integrated Experiments
FastAPI-based web dashboard showing live market data and signals
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import os

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Dashboard", description="Real-time Market Intelligence Dashboard")

# Setup directories
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Setup templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class DashboardData:
    """Manages dashboard data"""
    
    def __init__(self):
        self.data_file = Path("results/realtime_data/current_data.json")
        self.last_update = None
        self.cached_data = None
    
    def get_current_data(self) -> Dict[str, Any]:
        """Get current market data"""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                self.cached_data = data
                self.last_update = datetime.now()
                return data
            else:
                # Return mock data if no real data available
                return self._generate_mock_data()
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return self._generate_mock_data()
    
    def _generate_mock_data(self) -> Dict[str, Any]:
        """Generate mock data for demo purposes"""
        import numpy as np
        
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "7203.T", "SPY"]
        
        data = {
            "latest_prices": {},
            "technical_signals": {},
            "market_sentiment": {
                "overall_sentiment": "Bullish",
                "sentiment_score": 0.35,
                "news_volume": 150,
                "social_mentions": 750
            },
            "timestamp": datetime.now().isoformat()
        }
        
        for symbol in symbols:
            base_price = {"AAPL": 150, "GOOGL": 2800, "MSFT": 380, "TSLA": 250, 
                         "NVDA": 800, "7203.T": 2500, "SPY": 450}.get(symbol, 100)
            
            change_percent = np.random.normal(0, 1.5)
            price = base_price * (1 + change_percent / 100)
            
            data["latest_prices"][symbol] = {
                "price": round(price, 2),
                "change": round(price - base_price, 2),
                "change_percent": round(change_percent, 2),
                "volume": int(np.random.exponential(1000000)),
                "timestamp": datetime.now().isoformat()
            }
            
            signal = "BUY" if change_percent > 1 else "SELL" if change_percent < -1 else "HOLD"
            data["technical_signals"][symbol] = {
                "signal": signal,
                "confidence": round(np.random.uniform(0.5, 0.9), 2),
                "reason": f"Price change: {change_percent:.2f}%",
                "timestamp": datetime.now().isoformat()
            }
        
        return data

dashboard_data = DashboardData()

@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/data")
async def get_data():
    """API endpoint for current market data"""
    return dashboard_data.get_current_data()

@app.get("/api/charts/prices")
async def get_price_chart():
    """Generate price chart data"""
    data = dashboard_data.get_current_data()
    prices = data.get("latest_prices", {})
    
    symbols = list(prices.keys())
    price_values = [prices[symbol]["price"] for symbol in symbols]
    changes = [prices[symbol]["change_percent"] for symbol in symbols]
    
    # Create price chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=symbols,
        y=price_values,
        name="Current Price",
        marker_color=['green' if c >= 0 else 'red' for c in changes]
    ))
    
    fig.update_layout(
        title="Current Stock Prices",
        xaxis_title="Symbol",
        yaxis_title="Price ($)",
        height=400
    )
    
    return json.loads(fig.to_json())

@app.get("/api/charts/signals")
async def get_signals_chart():
    """Generate signals distribution chart"""
    data = dashboard_data.get_current_data()
    signals = data.get("technical_signals", {})
    
    signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for signal_data in signals.values():
        signal = signal_data.get("signal", "HOLD")
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
    
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=list(signal_counts.keys()),
        values=list(signal_counts.values()),
        marker_colors=['#4CAF50', '#f44336', '#FF9800']
    ))
    
    fig.update_layout(
        title="Signal Distribution",
        height=400
    )
    
    return json.loads(fig.to_json())

@app.get("/api/charts/performance")
async def get_performance_chart():
    """Generate performance metrics chart"""
    data = dashboard_data.get_current_data()
    prices = data.get("latest_prices", {})
    
    # Calculate performance metrics
    changes = [prices[symbol]["change_percent"] for symbol in prices.keys()]
    positive_count = len([c for c in changes if c > 0])
    negative_count = len([c for c in changes if c < 0])
    neutral_count = len(changes) - positive_count - negative_count
    
    avg_change = sum(changes) / len(changes) if changes else 0
    volatility = (sum((c - avg_change) ** 2 for c in changes) / len(changes)) ** 0.5 if len(changes) > 1 else 0
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Positive", "Negative", "Neutral", "Avg Change %", "Volatility %"],
        y=[positive_count, negative_count, neutral_count, avg_change, volatility],
        marker_color=['green', 'red', 'gray', 'blue', 'orange']
    ))
    
    fig.update_layout(
        title="Market Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=400
    )
    
    return json.loads(fig.to_json())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    try:
        while True:
            # Send current data
            data = dashboard_data.get_current_data()
            await websocket.send_json(data)
            
            # Wait 10 seconds before next update
            await asyncio.sleep(10)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

# Create dashboard HTML template
def create_dashboard_template():
    """Create the dashboard HTML template"""
    template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Real-time Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            margin: 0;
            color: #333;
            font-size: 2.5em;
            text-align: center;
        }
        
        .header .subtitle {
            text-align: center;
            color: #666;
            margin-top: 10px;
            font-size: 1.2em;
        }
        
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .status-item {
            text-align: center;
        }
        
        .status-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        .status-label {
            font-size: 0.9em;
            color: #666;
        }
        
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .wide-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .card h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .price-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .price-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .price-card.positive {
            border-left-color: #4CAF50;
            background: #f1f8e9;
        }
        
        .price-card.negative {
            border-left-color: #f44336;
            background: #ffebee;
        }
        
        .symbol {
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 5px;
        }
        
        .price {
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .change {
            font-size: 0.9em;
        }
        
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        .neutral { color: #666; }
        
        .signals-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
        }
        
        .signal-chip {
            padding: 10px 15px;
            border-radius: 25px;
            text-align: center;
            font-weight: bold;
            color: white;
        }
        
        .signal-buy { background: #4CAF50; }
        .signal-sell { background: #f44336; }
        .signal-hold { background: #FF9800; }
        
        .sentiment-indicator {
            text-align: center;
            padding: 20px;
        }
        
        .sentiment-score {
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .sentiment-label {
            font-size: 1.2em;
            margin-bottom: 10px;
        }
        
        .chart-container {
            height: 400px;
            margin-top: 20px;
        }
        
        .last-update {
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-style: italic;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .updating {
            animation: pulse 1s infinite;
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>🚀 MCP Real-time Market Dashboard</h1>
            <div class="subtitle">Comprehensive Market Intelligence Powered by MCP Servers</div>
            
            <div class="status-bar">
                <div class="status-item">
                    <div class="status-value" id="symbolCount">-</div>
                    <div class="status-label">Symbols Tracked</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="signalCount">-</div>
                    <div class="status-label">Active Signals</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="marketTrend">-</div>
                    <div class="status-label">Market Trend</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="lastUpdate">-</div>
                    <div class="status-label">Last Update</div>
                </div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📈 Current Prices</h3>
                <div class="price-grid" id="priceGrid">
                    <!-- Price cards will be inserted here -->
                </div>
            </div>
            
            <div class="card">
                <h3>🎯 Trading Signals</h3>
                <div class="signals-grid" id="signalsGrid">
                    <!-- Signal chips will be inserted here -->
                </div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📊 Price Chart</h3>
                <div class="chart-container" id="priceChart"></div>
            </div>
            
            <div class="card">
                <h3>🥧 Signal Distribution</h3>
                <div class="chart-container" id="signalsChart"></div>
            </div>
        </div>
        
        <div class="wide-grid">
            <div class="card">
                <h3>⚡ Performance Metrics</h3>
                <div class="chart-container" id="performanceChart"></div>
            </div>
        </div>
        
        <div class="card">
            <h3>📰 Market Sentiment</h3>
            <div class="sentiment-indicator">
                <div class="sentiment-score" id="sentimentScore">-</div>
                <div class="sentiment-label" id="sentimentLabel">Market Sentiment</div>
                <div id="sentimentDetails">
                    News Volume: <span id="newsVolume">-</span> | 
                    Social Mentions: <span id="socialMentions">-</span>
                </div>
            </div>
        </div>
        
        <div class="last-update">
            Last updated: <span id="lastUpdateTime">-</span>
        </div>
    </div>
    
    <script>
        // WebSocket connection for real-time updates
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
            // Fallback to polling
            setInterval(fetchData, 10000);
        };
        
        // Fallback data fetching
        async function fetchData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
        
        function updateDashboard(data) {
            updatePrices(data.latest_prices || {});
            updateSignals(data.technical_signals || {});
            updateSentiment(data.market_sentiment || {});
            updateStatusBar(data);
            updateCharts();
            
            // Update timestamp
            const timestamp = new Date(data.timestamp || Date.now()).toLocaleTimeString();
            document.getElementById('lastUpdateTime').textContent = timestamp;
        }
        
        function updatePrices(prices) {
            const grid = document.getElementById('priceGrid');
            grid.innerHTML = '';
            
            Object.entries(prices).forEach(([symbol, data]) => {
                const card = document.createElement('div');
                const changeClass = data.change_percent > 0 ? 'positive' : 
                                  data.change_percent < 0 ? 'negative' : 'neutral';
                
                card.className = `price-card ${changeClass}`;
                card.innerHTML = `
                    <div class="symbol">${symbol}</div>
                    <div class="price">$${data.price.toFixed(2)}</div>
                    <div class="change ${changeClass}">
                        ${data.change_percent > 0 ? '+' : ''}${data.change_percent.toFixed(2)}%
                    </div>
                `;
                grid.appendChild(card);
            });
        }
        
        function updateSignals(signals) {
            const grid = document.getElementById('signalsGrid');
            grid.innerHTML = '';
            
            Object.entries(signals).forEach(([symbol, data]) => {
                const chip = document.createElement('div');
                chip.className = `signal-chip signal-${data.signal.toLowerCase().replace('_', '-')}`;
                chip.innerHTML = `
                    <div>${symbol}</div>
                    <div>${data.signal}</div>
                    <div style="font-size: 0.8em;">${(data.confidence * 100).toFixed(0)}%</div>
                `;
                grid.appendChild(chip);
            });
        }
        
        function updateSentiment(sentiment) {
            const score = sentiment.sentiment_score || 0;
            const scoreElement = document.getElementById('sentimentScore');
            const labelElement = document.getElementById('sentimentLabel');
            
            scoreElement.textContent = score.toFixed(2);
            scoreElement.className = 'sentiment-score ' + 
                (score > 0.1 ? 'positive' : score < -0.1 ? 'negative' : 'neutral');
            
            labelElement.textContent = sentiment.overall_sentiment || 'Neutral';
            
            document.getElementById('newsVolume').textContent = sentiment.news_volume || 0;
            document.getElementById('socialMentions').textContent = sentiment.social_mentions || 0;
        }
        
        function updateStatusBar(data) {
            const prices = data.latest_prices || {};
            const signals = data.technical_signals || {};
            
            document.getElementById('symbolCount').textContent = Object.keys(prices).length;
            document.getElementById('signalCount').textContent = Object.keys(signals).length;
            
            // Calculate market trend
            const changes = Object.values(prices).map(p => p.change_percent);
            const avgChange = changes.reduce((a, b) => a + b, 0) / changes.length;
            const trend = avgChange > 0.5 ? '📈 Up' : avgChange < -0.5 ? '📉 Down' : '➡️ Flat';
            document.getElementById('marketTrend').textContent = trend;
            
            const now = new Date();
            document.getElementById('lastUpdate').textContent = now.toLocaleTimeString();
        }
        
        async function updateCharts() {
            try {
                // Update price chart
                const priceResponse = await fetch('/api/charts/prices');
                const priceChart = await priceResponse.json();
                Plotly.newPlot('priceChart', priceChart.data, priceChart.layout);
                
                // Update signals chart
                const signalsResponse = await fetch('/api/charts/signals');
                const signalsChart = await signalsResponse.json();
                Plotly.newPlot('signalsChart', signalsChart.data, signalsChart.layout);
                
                // Update performance chart
                const performanceResponse = await fetch('/api/charts/performance');
                const performanceChart = await performanceResponse.json();
                Plotly.newPlot('performanceChart', performanceChart.data, performanceChart.layout);
                
            } catch (error) {
                console.error('Error updating charts:', error);
            }
        }
        
        // Initial load
        fetchData();
    </script>
</body>
</html>
"""
    
    template_file = TEMPLATES_DIR / "dashboard.html"
    with open(template_file, 'w') as f:
        f.write(template_content)

# Create template on startup
create_dashboard_template()

if __name__ == "__main__":
    print("🚀 Starting MCP Real-time Dashboard...")
    print("Dashboard will be available at: http://localhost:8080")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(
        "dashboard_app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )