#!/usr/bin/env python3
"""
MCP Integrated Experiment Framework
Conducts comprehensive market analysis experiments using all MCP servers
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add MCP servers to path
sys.path.append(str(Path(__file__).parent.parent / "mcp_servers"))

from mcp_servers.server_manager import MCPServerManager, MCPConfig
from mcp_servers.finance_server import FinanceServer, FinanceConfig
from mcp_servers.macro_data_server import MacroDataServer, MacroDataConfig
from mcp_servers.alternative_data_server import AlternativeDataServer, AlternativeDataConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for MCP integrated experiments"""
    # Experiment settings
    experiment_name: str = "mcp_integrated_analysis"
    duration_hours: int = 24
    data_collection_interval_minutes: int = 30
    
    # Stock universe
    target_symbols: List[str] = field(default_factory=lambda: [
        # US Large Cap
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V",
        # Japanese Stocks
        "7203.T", "6758.T", "9984.T", "7974.T", "4063.T", "6861.T", "8306.T", "9432.T",
        # Market ETFs
        "SPY", "QQQ", "IWM", "VTI", "1306.T"
    ])
    
    # Data sources to include
    include_finance_data: bool = True
    include_macro_data: bool = True
    include_alternative_data: bool = True
    
    # Analysis settings
    generate_signals: bool = True
    calculate_correlations: bool = True
    sentiment_analysis: bool = True
    
    # Output settings
    save_raw_data: bool = True
    save_processed_data: bool = True
    generate_dashboard: bool = True
    
    # Performance settings
    max_concurrent_requests: int = 10
    timeout_seconds: int = 30
    retry_attempts: int = 3

class MCPIntegratedExperiment:
    """Comprehensive market analysis experiment using MCP servers"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.start_time = datetime.now()
        self.experiment_id = f"{config.experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Data storage
        self.raw_data = {}
        self.processed_data = {}
        self.signals = {}
        self.metrics = {
            "data_points_collected": 0,
            "api_calls_made": 0,
            "errors_encountered": 0,
            "signals_generated": 0,
            "experiment_start": self.start_time.isoformat(),
            "data_collection_rounds": 0
        }
        
        # Initialize MCP servers
        self._initialize_mcp_servers()
        
        # Create output directories
        self.output_dir = Path(f"results/mcp_experiments/{self.experiment_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized MCP Integrated Experiment: {self.experiment_id}")
    
    def _initialize_mcp_servers(self):
        """Initialize MCP servers"""
        try:
            # Create MCP configuration
            mcp_config = MCPConfig.from_env()
            
            # Initialize individual servers (fallback if manager not available)
            if self.config.include_finance_data:
                finance_config = FinanceConfig(
                    yahoo_finance_enabled=True,
                    cache_duration_minutes=5  # Short cache for experiments
                )
                self.finance_server = FinanceServer(finance_config)
                logger.info("Finance server initialized")
            
            if self.config.include_macro_data:
                macro_config = MacroDataConfig(
                    fred_api_key=os.getenv("FRED_API_KEY"),
                    cache_duration_hours=1
                )
                self.macro_server = MacroDataServer(macro_config)
                logger.info("Macro data server initialized")
            
            if self.config.include_alternative_data:
                alt_config = AlternativeDataConfig(
                    news_api_key=os.getenv("NEWS_API_KEY"),
                    cache_duration_hours=1
                )
                self.alternative_server = AlternativeDataServer(alt_config)
                logger.info("Alternative data server initialized")
                
        except Exception as e:
            logger.error(f"Error initializing MCP servers: {str(e)}")
            raise
    
    async def run_experiment(self):
        """Run the complete experiment"""
        logger.info(f"Starting MCP Integrated Experiment: {self.experiment_id}")
        
        try:
            # Initial data collection
            await self._collect_initial_data()
            
            # Run periodic data collection
            await self._run_periodic_collection()
            
            # Generate comprehensive analysis
            await self._generate_analysis()
            
            # Save results
            await self._save_results()
            
            # Generate dashboard
            if self.config.generate_dashboard:
                await self._generate_dashboard()
            
            logger.info("Experiment completed successfully")
            return self._get_experiment_summary()
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            self.metrics["errors_encountered"] += 1
            raise
    
    async def _collect_initial_data(self):
        """Collect initial baseline data"""
        logger.info("Collecting initial baseline data...")
        
        tasks = []
        
        # Collect finance data for all symbols
        if self.config.include_finance_data:
            for symbol in self.config.target_symbols:
                tasks.append(self._collect_finance_data(symbol))
        
        # Collect macro economic context
        if self.config.include_macro_data:
            tasks.append(self._collect_macro_data())
        
        # Collect alternative data insights
        if self.config.include_alternative_data:
            tasks.append(self._collect_alternative_data())
        
        # Execute all data collection tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Data collection task {i} failed: {str(result)}")
                self.metrics["errors_encountered"] += 1
            else:
                self.metrics["data_points_collected"] += 1
        
        self.metrics["data_collection_rounds"] += 1
        logger.info(f"Initial data collection completed. Collected {self.metrics['data_points_collected']} data points")
    
    async def _run_periodic_collection(self):
        """Run periodic data collection"""
        logger.info("Starting periodic data collection...")
        
        end_time = self.start_time + timedelta(hours=self.config.duration_hours)
        interval = timedelta(minutes=self.config.data_collection_interval_minutes)
        
        next_collection = self.start_time + interval
        
        while datetime.now() < end_time:
            # Wait until next collection time
            now = datetime.now()
            if now < next_collection:
                sleep_time = (next_collection - now).total_seconds()
                await asyncio.sleep(sleep_time)
            
            # Collect data
            try:
                await self._collect_periodic_data()
                self.metrics["data_collection_rounds"] += 1
                
                # Generate signals if enabled
                if self.config.generate_signals:
                    await self._generate_periodic_signals()
                
                logger.info(f"Periodic collection round {self.metrics['data_collection_rounds']} completed")
                
            except Exception as e:
                logger.error(f"Periodic collection failed: {str(e)}")
                self.metrics["errors_encountered"] += 1
            
            # Schedule next collection
            next_collection += interval
        
        logger.info("Periodic data collection completed")
    
    async def _collect_finance_data(self, symbol: str):
        """Collect financial data for a symbol"""
        try:
            # Get current price and basic info
            price_data = await self.finance_server._get_stock_price({
                "symbol": symbol,
                "period": "1d",
                "interval": "1m"
            })
            
            # Get technical indicators
            tech_data = await self.finance_server._calculate_technical_indicators({
                "symbol": symbol,
                "indicators": ["sma", "rsi", "macd"],
                "period": "1mo"
            })
            
            # Store data
            if symbol not in self.raw_data:
                self.raw_data[symbol] = {"finance": []}
            
            self.raw_data[symbol]["finance"].append({
                "timestamp": datetime.now().isoformat(),
                "price_data": price_data[0].data if price_data else None,
                "technical_data": tech_data[0].data if tech_data else None
            })
            
            self.metrics["api_calls_made"] += 2
            
        except Exception as e:
            logger.warning(f"Failed to collect finance data for {symbol}: {str(e)}")
            raise
    
    async def _collect_macro_data(self):
        """Collect macro economic data"""
        try:
            # Get key economic indicators
            indicators_data = await self.macro_server._get_key_indicators({
                "country": "us",
                "category": "overview"
            })
            
            # Store data
            if "macro" not in self.raw_data:
                self.raw_data["macro"] = []
            
            self.raw_data["macro"].append({
                "timestamp": datetime.now().isoformat(),
                "indicators": indicators_data[0].data if indicators_data else None
            })
            
            self.metrics["api_calls_made"] += 1
            
        except Exception as e:
            logger.warning(f"Failed to collect macro data: {str(e)}")
            raise
    
    async def _collect_alternative_data(self):
        """Collect alternative data"""
        try:
            # Collect market sentiment for key symbols
            sentiment_tasks = []
            key_symbols = self.config.target_symbols[:5]  # Limit to avoid rate limits
            
            for symbol in key_symbols:
                sentiment_tasks.append(
                    self.alternative_server._get_news_sentiment({
                        "query": symbol,
                        "time_period": "1d",
                        "sentiment_analysis": True
                    })
                )
            
            sentiment_results = await asyncio.gather(*sentiment_tasks, return_exceptions=True)
            
            # Store data
            if "alternative" not in self.raw_data:
                self.raw_data["alternative"] = []
            
            alternative_data = {
                "timestamp": datetime.now().isoformat(),
                "sentiment_data": {}
            }
            
            for i, result in enumerate(sentiment_results):
                if not isinstance(result, Exception) and result:
                    alternative_data["sentiment_data"][key_symbols[i]] = result[0].data
            
            self.raw_data["alternative"].append(alternative_data)
            self.metrics["api_calls_made"] += len(key_symbols)
            
        except Exception as e:
            logger.warning(f"Failed to collect alternative data: {str(e)}")
            raise
    
    async def _collect_periodic_data(self):
        """Collect data during periodic rounds"""
        # Collect data for a subset of symbols to manage API limits
        active_symbols = self.config.target_symbols[:10]  # Limit for periodic collection
        
        tasks = []
        
        # Collect updated finance data
        if self.config.include_finance_data:
            for symbol in active_symbols:
                tasks.append(self._collect_finance_data(symbol))
        
        # Collect updated macro data (less frequent)
        if self.config.include_macro_data and self.metrics["data_collection_rounds"] % 4 == 0:
            tasks.append(self._collect_macro_data())
        
        # Collect updated alternative data
        if self.config.include_alternative_data:
            tasks.append(self._collect_alternative_data())
        
        # Execute collection tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful collections
        successful = sum(1 for r in results if not isinstance(r, Exception))
        self.metrics["data_points_collected"] += successful
    
    async def _generate_periodic_signals(self):
        """Generate investment signals from current data"""
        try:
            signals = {}
            
            for symbol in self.config.target_symbols[:10]:  # Limit processing
                if symbol in self.raw_data and "finance" in self.raw_data[symbol]:
                    # Get latest financial data
                    latest_finance = self.raw_data[symbol]["finance"][-1] if self.raw_data[symbol]["finance"] else None
                    
                    if latest_finance and latest_finance.get("technical_data"):
                        signal = self._calculate_simple_signal(symbol, latest_finance)
                        signals[symbol] = signal
                        self.metrics["signals_generated"] += 1
            
            # Store signals with timestamp
            timestamp = datetime.now().isoformat()
            if "signals" not in self.processed_data:
                self.processed_data["signals"] = []
            
            self.processed_data["signals"].append({
                "timestamp": timestamp,
                "signals": signals
            })
            
        except Exception as e:
            logger.warning(f"Failed to generate signals: {str(e)}")
    
    def _calculate_simple_signal(self, symbol: str, finance_data: Dict) -> Dict[str, Any]:
        """Calculate a simple trading signal"""
        try:
            tech_data = finance_data.get("technical_data", {})
            indicators = tech_data.get("indicators", {})
            
            # Simple signal based on RSI and SMA
            rsi = indicators.get("rsi", [])[-1] if indicators.get("rsi") else 50
            sma_20 = indicators.get("sma_20", [])[-1] if indicators.get("sma_20") else None
            sma_50 = indicators.get("sma_50", [])[-1] if indicators.get("sma_50") else None
            
            # Generate signal
            signal = "HOLD"
            confidence = 0.5
            
            if rsi < 30 and sma_20 and sma_50 and sma_20 > sma_50:
                signal = "BUY"
                confidence = 0.7
            elif rsi > 70 and sma_20 and sma_50 and sma_20 < sma_50:
                signal = "SELL"
                confidence = 0.7
            elif rsi < 40 and sma_20 and sma_50 and sma_20 > sma_50:
                signal = "BUY"
                confidence = 0.6
            elif rsi > 60 and sma_20 and sma_50 and sma_20 < sma_50:
                signal = "SELL"
                confidence = 0.6
            
            return {
                "signal": signal,
                "confidence": confidence,
                "rsi": rsi,
                "sma_trend": "UP" if sma_20 and sma_50 and sma_20 > sma_50 else "DOWN" if sma_20 and sma_50 else "NEUTRAL"
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate signal for {symbol}: {str(e)}")
            return {"signal": "HOLD", "confidence": 0.5, "error": str(e)}
    
    async def _generate_analysis(self):
        """Generate comprehensive analysis from collected data"""
        logger.info("Generating comprehensive analysis...")
        
        try:
            # Calculate correlations if enabled
            if self.config.calculate_correlations:
                self._calculate_price_correlations()
            
            # Analyze sentiment trends
            if self.config.sentiment_analysis:
                self._analyze_sentiment_trends()
            
            # Generate market summary
            self._generate_market_summary()
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
            
            logger.info("Comprehensive analysis completed")
            
        except Exception as e:
            logger.error(f"Analysis generation failed: {str(e)}")
            self.metrics["errors_encountered"] += 1
    
    def _calculate_price_correlations(self):
        """Calculate price correlations between symbols"""
        try:
            price_data = {}
            
            # Extract price data for correlation calculation
            for symbol in self.config.target_symbols:
                if symbol in self.raw_data and "finance" in self.raw_data[symbol]:
                    prices = []
                    for data_point in self.raw_data[symbol]["finance"]:
                        if data_point.get("price_data") and data_point["price_data"].get("data"):
                            # Get latest price from the data
                            latest_price = data_point["price_data"]["data"][-1].get("Close") if data_point["price_data"]["data"] else None
                            if latest_price:
                                prices.append(latest_price)
                    
                    if prices:
                        price_data[symbol] = prices
            
            # Calculate correlation matrix
            if len(price_data) > 1:
                df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in price_data.items()]))
                correlation_matrix = df.corr().to_dict()
                
                self.processed_data["correlations"] = {
                    "matrix": correlation_matrix,
                    "timestamp": datetime.now().isoformat(),
                    "symbols_included": list(price_data.keys())
                }
            
        except Exception as e:
            logger.warning(f"Failed to calculate correlations: {str(e)}")
    
    def _analyze_sentiment_trends(self):
        """Analyze sentiment trends from alternative data"""
        try:
            if "alternative" not in self.raw_data:
                return
            
            sentiment_trends = {}
            
            for data_point in self.raw_data["alternative"]:
                if "sentiment_data" in data_point:
                    for symbol, sentiment_info in data_point["sentiment_data"].items():
                        if symbol not in sentiment_trends:
                            sentiment_trends[symbol] = []
                        
                        if sentiment_info and "summary" in sentiment_info:
                            summary = sentiment_info["summary"]
                            sentiment_score = summary.get("sentiment_analysis", {}).get("polarity", 0)
                            sentiment_trends[symbol].append({
                                "timestamp": data_point["timestamp"],
                                "sentiment_score": sentiment_score,
                                "article_count": summary.get("total_articles", 0)
                            })
            
            self.processed_data["sentiment_trends"] = sentiment_trends
            
        except Exception as e:
            logger.warning(f"Failed to analyze sentiment trends: {str(e)}")
    
    def _generate_market_summary(self):
        """Generate overall market summary"""
        try:
            summary = {
                "experiment_id": self.experiment_id,
                "symbols_analyzed": len(self.config.target_symbols),
                "data_collection_rounds": self.metrics["data_collection_rounds"],
                "total_data_points": self.metrics["data_points_collected"],
                "signals_generated": self.metrics["signals_generated"],
                "experiment_duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add latest signals summary
            if "signals" in self.processed_data and self.processed_data["signals"]:
                latest_signals = self.processed_data["signals"][-1]["signals"]
                signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
                
                for signal_data in latest_signals.values():
                    signal = signal_data.get("signal", "HOLD")
                    signal_counts[signal] = signal_counts.get(signal, 0) + 1
                
                summary["latest_signal_distribution"] = signal_counts
            
            self.processed_data["market_summary"] = summary
            
        except Exception as e:
            logger.warning(f"Failed to generate market summary: {str(e)}")
    
    def _calculate_performance_metrics(self):
        """Calculate experiment performance metrics"""
        try:
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            performance = {
                "data_collection_rate": self.metrics["data_points_collected"] / max(duration / 3600, 0.1),  # per hour
                "api_call_rate": self.metrics["api_calls_made"] / max(duration / 3600, 0.1),  # per hour
                "error_rate": self.metrics["errors_encountered"] / max(self.metrics["api_calls_made"], 1),
                "success_rate": 1 - (self.metrics["errors_encountered"] / max(self.metrics["api_calls_made"], 1)),
                "signals_per_round": self.metrics["signals_generated"] / max(self.metrics["data_collection_rounds"], 1)
            }
            
            self.processed_data["performance_metrics"] = performance
            
        except Exception as e:
            logger.warning(f"Failed to calculate performance metrics: {str(e)}")
    
    async def _save_results(self):
        """Save experiment results"""
        logger.info("Saving experiment results...")
        
        try:
            # Save raw data
            if self.config.save_raw_data:
                raw_data_file = self.output_dir / "raw_data.json"
                with open(raw_data_file, 'w') as f:
                    json.dump(self.raw_data, f, indent=2, default=str)
            
            # Save processed data
            if self.config.save_processed_data:
                processed_data_file = self.output_dir / "processed_data.json"
                with open(processed_data_file, 'w') as f:
                    json.dump(self.processed_data, f, indent=2, default=str)
            
            # Save metrics
            metrics_file = self.output_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            
            # Save experiment configuration
            config_file = self.output_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)
            
            logger.info(f"Results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise
    
    async def _generate_dashboard(self):
        """Generate HTML dashboard for experiment results"""
        logger.info("Generating dashboard...")
        
        try:
            dashboard_html = self._create_dashboard_html()
            
            dashboard_file = self.output_dir / "dashboard.html"
            with open(dashboard_file, 'w') as f:
                f.write(dashboard_html)
            
            logger.info(f"Dashboard generated: {dashboard_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard: {str(e)}")
    
    def _create_dashboard_html(self) -> str:
        """Create HTML dashboard content"""
        # Get latest data for dashboard
        market_summary = self.processed_data.get("market_summary", {})
        performance_metrics = self.processed_data.get("performance_metrics", {})
        latest_signals = {}
        
        if "signals" in self.processed_data and self.processed_data["signals"]:
            latest_signals = self.processed_data["signals"][-1].get("signals", {})
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Integrated Experiment Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .dashboard-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .metrics-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #666; font-size: 0.9em; margin-top: 5px; }}
        .signals-container {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .signal-item {{ display: inline-block; margin: 5px; padding: 10px 15px; border-radius: 20px; font-weight: bold; }}
        .signal-buy {{ background-color: #4CAF50; color: white; }}
        .signal-sell {{ background-color: #f44336; color: white; }}
        .signal-hold {{ background-color: #FF9800; color: white; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .footer {{ text-align: center; color: #666; margin-top: 40px; }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>🚀 MCP Integrated Experiment Dashboard</h1>
        <h2>{self.experiment_id}</h2>
        <p>Real-time market intelligence powered by comprehensive data sources</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="metrics-container">
        <div class="metric-card">
            <div class="metric-value">{market_summary.get('symbols_analyzed', 0)}</div>
            <div class="metric-label">Symbols Analyzed</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{market_summary.get('data_collection_rounds', 0)}</div>
            <div class="metric-label">Data Collection Rounds</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{market_summary.get('total_data_points', 0)}</div>
            <div class="metric-label">Total Data Points</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{market_summary.get('signals_generated', 0)}</div>
            <div class="metric-label">Signals Generated</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{performance_metrics.get('success_rate', 0):.1%}</div>
            <div class="metric-label">Success Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{performance_metrics.get('data_collection_rate', 0):.1f}</div>
            <div class="metric-label">Data Points/Hour</div>
        </div>
    </div>
    
    <div class="signals-container">
        <h3>🎯 Latest Investment Signals</h3>
        <p><strong>Signal Distribution:</strong> 
            BUY: {market_summary.get('latest_signal_distribution', {}).get('BUY', 0)} | 
            SELL: {market_summary.get('latest_signal_distribution', {}).get('SELL', 0)} | 
            HOLD: {market_summary.get('latest_signal_distribution', {}).get('HOLD', 0)}
        </p>
        <div class="signals-grid">
"""
        
        # Add individual signals
        for symbol, signal_data in list(latest_signals.items())[:20]:  # Limit display
            signal = signal_data.get('signal', 'HOLD')
            confidence = signal_data.get('confidence', 0.5)
            css_class = f"signal-{signal.lower()}"
            html += f'<div class="signal-item {css_class}">{symbol}: {signal} ({confidence:.1%})</div>'
        
        html += """
        </div>
    </div>
    
    <div class="chart-container">
        <h3>📊 Signal Distribution Over Time</h3>
        <div id="signalsChart" style="height: 400px;"></div>
    </div>
    
    <div class="chart-container">
        <h3>⚡ Performance Metrics</h3>
        <div id="performanceChart" style="height: 400px;"></div>
    </div>
    
    <div class="footer">
        <p>📈 Powered by Alpha Architecture Agent MCP Servers</p>
        <p>Comprehensive market intelligence through integrated data sources</p>
    </div>
    
    <script>
        // Signal distribution chart
        const signalData = [{
            x: ['BUY', 'SELL', 'HOLD'],
            y: [""" + str([
                market_summary.get('latest_signal_distribution', {}).get('BUY', 0),
                market_summary.get('latest_signal_distribution', {}).get('SELL', 0),
                market_summary.get('latest_signal_distribution', {}).get('HOLD', 0)
            ]).strip('[]') + """],
            type: 'bar',
            marker: { color: ['#4CAF50', '#f44336', '#FF9800'] }
        }];
        
        Plotly.newPlot('signalsChart', signalData, {
            title: 'Current Signal Distribution',
            xaxis: { title: 'Signal Type' },
            yaxis: { title: 'Count' }
        });
        
        // Performance metrics chart
        const performanceData = [{
            x: ['Success Rate', 'Data Collection Rate', 'API Call Rate'],
            y: [""" + str([
                performance_metrics.get('success_rate', 0) * 100,
                performance_metrics.get('data_collection_rate', 0),
                performance_metrics.get('api_call_rate', 0)
            ]).strip('[]') + """],
            type: 'bar',
            marker: { color: '#667eea' }
        }];
        
        Plotly.newPlot('performanceChart', performanceData, {
            title: 'Performance Metrics',
            xaxis: { title: 'Metric' },
            yaxis: { title: 'Value' }
        });
        
        // Auto-refresh every 5 minutes
        setTimeout(() => {
            location.reload();
        }, 300000);
    </script>
</body>
</html>
"""
        return html
    
    def _get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary"""
        return {
            "experiment_id": self.experiment_id,
            "status": "completed",
            "duration": (datetime.now() - self.start_time).total_seconds(),
            "metrics": self.metrics,
            "output_directory": str(self.output_dir),
            "dashboard_url": f"file://{self.output_dir / 'dashboard.html'}"
        }

async def main():
    """Main experiment execution"""
    # Configure experiment
    config = ExperimentConfig(
        experiment_name="mcp_comprehensive_analysis",
        duration_hours=2,  # Short duration for testing
        data_collection_interval_minutes=15,
        target_symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "7203.T", "SPY"],
        generate_signals=True,
        calculate_correlations=True,
        sentiment_analysis=True
    )
    
    # Run experiment
    experiment = MCPIntegratedExperiment(config)
    summary = await experiment.run_experiment()
    
    print("\n" + "="*60)
    print("🎉 MCP INTEGRATED EXPERIMENT COMPLETED!")
    print("="*60)
    print(f"Experiment ID: {summary['experiment_id']}")
    print(f"Duration: {summary['duration']:.1f} seconds")
    print(f"Data Points Collected: {summary['metrics']['data_points_collected']}")
    print(f"Signals Generated: {summary['metrics']['signals_generated']}")
    print(f"Success Rate: {summary['metrics'].get('success_rate', 0):.1%}")
    print(f"Dashboard: {summary['dashboard_url']}")
    print("="*60)
    
    return summary

if __name__ == "__main__":
    asyncio.run(main())