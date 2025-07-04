#!/usr/bin/env python3
"""
Real-time Data Collector for MCP Integrated Experiments
Continuously collects data from all MCP servers and feeds the dashboard
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path
import os
import sys

# Add MCP servers to path
sys.path.append(str(Path(__file__).parent.parent / "mcp_servers"))

import yfinance as yf
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeDataCollector:
    """Collects real-time market data for dashboard display"""
    
    def __init__(self):
        self.symbols = [
            # US Tech Giants
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
            # Japanese Stocks
            "7203.T", "6758.T", "9984.T", "7974.T",
            # ETFs
            "SPY", "QQQ", "VTI"
        ]
        
        self.data_store = {
            "latest_prices": {},
            "price_changes": {},
            "technical_signals": {},
            "market_sentiment": {},
            "economic_indicators": {},
            "timestamp": None
        }
        
        self.output_dir = Path("results/realtime_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def start_collection(self, duration_minutes: int = 60):
        """Start real-time data collection"""
        logger.info(f"Starting real-time data collection for {duration_minutes} minutes")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time:
            try:
                # Collect current market data
                await self._collect_market_data()
                
                # Calculate signals
                self._calculate_signals()
                
                # Save data
                self._save_current_data()
                
                # Log progress
                logger.info(f"Data collected at {datetime.now().strftime('%H:%M:%S')}")
                
                # Wait for next collection (30 seconds)
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in data collection: {str(e)}")
                await asyncio.sleep(5)
        
        logger.info("Real-time data collection completed")
    
    async def _collect_market_data(self):
        """Collect current market data"""
        try:
            prices = {}
            changes = {}
            
            # Use yfinance for real-time data
            tickers = yf.Tickers(' '.join(self.symbols))
            
            for symbol in self.symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    hist = ticker.history(period="2d", interval="1m")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        previous_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        change = current_price - previous_price
                        change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
                        
                        prices[symbol] = {
                            "price": float(current_price),
                            "change": float(change),
                            "change_percent": float(change_percent),
                            "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Store for signal calculation
                        changes[symbol] = hist['Close'].pct_change().dropna()
                        
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {str(e)}")
                    continue
            
            self.data_store["latest_prices"] = prices
            self.data_store["price_changes"] = changes
            self.data_store["timestamp"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error collecting market data: {str(e)}")
    
    def _calculate_signals(self):
        """Calculate trading signals from current data"""
        signals = {}
        
        for symbol, price_data in self.data_store["latest_prices"].items():
            try:
                # Simple momentum signal
                change_percent = price_data.get("change_percent", 0)
                volume = price_data.get("volume", 0)
                
                # Signal logic
                if change_percent > 2 and volume > 1000000:
                    signal = "STRONG_BUY"
                    confidence = 0.8
                elif change_percent > 1:
                    signal = "BUY"
                    confidence = 0.6
                elif change_percent < -2 and volume > 1000000:
                    signal = "STRONG_SELL"
                    confidence = 0.8
                elif change_percent < -1:
                    signal = "SELL"
                    confidence = 0.6
                else:
                    signal = "HOLD"
                    confidence = 0.5
                
                # Add volatility factor
                if symbol in self.data_store["price_changes"]:
                    returns = self.data_store["price_changes"][symbol]
                    if len(returns) > 10:
                        volatility = returns.std() * np.sqrt(252)  # Annualized
                        if volatility > 0.5:  # High volatility
                            confidence *= 0.8
                
                signals[symbol] = {
                    "signal": signal,
                    "confidence": confidence,
                    "reason": f"Price change: {change_percent:.2f}%, Volume: {volume:,}",
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.warning(f"Failed to calculate signal for {symbol}: {str(e)}")
                signals[symbol] = {
                    "signal": "HOLD",
                    "confidence": 0.5,
                    "reason": "Calculation error",
                    "error": str(e)
                }
        
        self.data_store["technical_signals"] = signals
    
    def _save_current_data(self):
        """Save current data to file for dashboard"""
        try:
            # Save to JSON file for dashboard consumption
            data_file = self.output_dir / "current_data.json"
            with open(data_file, 'w') as f:
                json.dump(self.data_store, f, indent=2, default=str)
            
            # Also save timestamped file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_file = self.output_dir / f"data_{timestamp}.json"
            with open(timestamped_file, 'w') as f:
                json.dump(self.data_store, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")

class MockDataGenerator:
    """Generate mock data when real APIs are not available"""
    
    def __init__(self):
        self.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "7203.T", "SPY"]
        self.base_prices = {
            "AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 380.0, "TSLA": 250.0,
            "NVDA": 800.0, "7203.T": 2500.0, "SPY": 450.0
        }
        
    def generate_mock_data(self) -> Dict[str, Any]:
        """Generate realistic mock market data"""
        data = {
            "latest_prices": {},
            "technical_signals": {},
            "market_sentiment": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate price data
        for symbol in self.symbols:
            base_price = self.base_prices.get(symbol, 100.0)
            
            # Add random price movement
            change_percent = np.random.normal(0, 2)  # Mean 0, std 2%
            new_price = base_price * (1 + change_percent / 100)
            change = new_price - base_price
            volume = int(np.random.exponential(1000000))
            
            data["latest_prices"][symbol] = {
                "price": round(new_price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "volume": volume,
                "timestamp": datetime.now().isoformat()
            }
            
            # Generate signals
            if change_percent > 1.5:
                signal = "BUY"
                confidence = 0.7
            elif change_percent < -1.5:
                signal = "SELL"
                confidence = 0.7
            else:
                signal = "HOLD"
                confidence = 0.5
                
            data["technical_signals"][symbol] = {
                "signal": signal,
                "confidence": confidence,
                "reason": f"Price change: {change_percent:.2f}%",
                "timestamp": datetime.now().isoformat()
            }
        
        # Generate market sentiment
        data["market_sentiment"] = {
            "overall_sentiment": np.random.choice(["Bullish", "Bearish", "Neutral"], p=[0.4, 0.3, 0.3]),
            "sentiment_score": round(np.random.uniform(-1, 1), 2),
            "news_volume": int(np.random.exponential(100)),
            "social_mentions": int(np.random.exponential(500))
        }
        
        # Update base prices for next iteration
        for symbol in self.symbols:
            if symbol in data["latest_prices"]:
                self.base_prices[symbol] = data["latest_prices"][symbol]["price"]
        
        return data

async def run_mock_data_collection():
    """Run mock data collection for testing"""
    logger.info("Starting mock data collection...")
    
    generator = MockDataGenerator()
    output_dir = Path("results/realtime_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run for 30 iterations (15 minutes with 30-second intervals)
    for i in range(30):
        try:
            # Generate mock data
            data = generator.generate_mock_data()
            
            # Save data
            data_file = output_dir / "current_data.json"
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Save timestamped file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_file = output_dir / f"mock_data_{timestamp}.json"
            with open(timestamped_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Mock data generated: iteration {i+1}/30")
            
            # Wait 30 seconds
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Error in mock data generation: {str(e)}")
            await asyncio.sleep(5)
    
    logger.info("Mock data collection completed")

async def main():
    """Main data collection runner"""
    print("Select data collection mode:")
    print("1. Real-time data collection (requires API access)")
    print("2. Mock data generation (for testing)")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        collector = RealTimeDataCollector()
        await collector.start_collection(duration_minutes=60)
    elif choice == "2":
        await run_mock_data_collection()
    else:
        print("Invalid choice. Running mock data generation...")
        await run_mock_data_collection()

if __name__ == "__main__":
    asyncio.run(main())