#!/usr/bin/env python3
"""
Experiment Runner - Orchestrates the complete MCP experiment workflow
Runs data collection, starts dashboard, and monitors the experiment
"""
import asyncio
import subprocess
import time
import signal
import sys
import os
from pathlib import Path
from datetime import datetime
import threading
import webbrowser

# Setup paths
EXPERIMENT_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENT_DIR.parent

class ExperimentRunner:
    """Orchestrates the complete experiment workflow"""
    
    def __init__(self):
        self.processes = []
        self.data_collector_process = None
        self.dashboard_process = None
        self.experiment_start_time = datetime.now()
        
        # Create results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        print("🚀 Alpha Architecture Agent - MCP Integrated Experiment")
        print("=" * 60)
        print(f"Experiment Start Time: {self.experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print("\n⚠️  Received shutdown signal. Stopping experiment...")
            self.stop_experiment()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_data_collector(self):
        """Start the real-time data collector"""
        print("📊 Starting real-time data collector...")
        
        try:
            # Start data collector in background
            cmd = [sys.executable, str(EXPERIMENT_DIR / "realtime_data_collector.py")]
            
            # Create a process that will run mock data generation
            self.data_collector_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True
            )
            
            # Send "2" to select mock data generation
            self.data_collector_process.stdin.write("2\n")
            self.data_collector_process.stdin.flush()
            
            print("✅ Data collector started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start data collector: {str(e)}")
            return False
    
    def start_dashboard(self):
        """Start the web dashboard"""
        print("🌐 Starting web dashboard...")
        
        try:
            # Start dashboard server
            cmd = [sys.executable, str(EXPERIMENT_DIR / "dashboard_app.py")]
            
            self.dashboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for server to start
            time.sleep(3)
            
            print("✅ Dashboard started successfully")
            print("🌐 Dashboard URL: http://localhost:8080")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start dashboard: {str(e)}")
            return False
    
    def open_dashboard(self):
        """Open dashboard in web browser"""
        try:
            print("🔗 Opening dashboard in web browser...")
            webbrowser.open("http://localhost:8080")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {str(e)}")
            print("   Please manually open: http://localhost:8080")
    
    def monitor_experiment(self, duration_minutes: int = 30):
        """Monitor the experiment and display status"""
        print(f"⏱️  Monitoring experiment for {duration_minutes} minutes...")
        print("   Press Ctrl+C to stop early")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        try:
            while time.time() < end_time:
                elapsed = time.time() - start_time
                remaining = end_time - time.time()
                
                # Check if processes are still running
                data_status = "🟢 Running" if self.data_collector_process and self.data_collector_process.poll() is None else "🔴 Stopped"
                dashboard_status = "🟢 Running" if self.dashboard_process and self.dashboard_process.poll() is None else "🔴 Stopped"
                
                print(f"\r⏰ Elapsed: {int(elapsed//60):02d}:{int(elapsed%60):02d} | "
                      f"Remaining: {int(remaining//60):02d}:{int(remaining%60):02d} | "
                      f"Data Collector: {data_status} | "
                      f"Dashboard: {dashboard_status}", end="", flush=True)
                
                time.sleep(10)  # Update every 10 seconds
            
            print("\n⏰ Experiment duration completed!")
            
        except KeyboardInterrupt:
            print("\n⚠️  Experiment interrupted by user")
    
    def stop_experiment(self):
        """Stop all experiment processes"""
        print("\n🛑 Stopping experiment processes...")
        
        if self.data_collector_process:
            try:
                self.data_collector_process.terminate()
                self.data_collector_process.wait(timeout=5)
                print("✅ Data collector stopped")
            except:
                self.data_collector_process.kill()
                print("🔥 Data collector force killed")
        
        if self.dashboard_process:
            try:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=5)
                print("✅ Dashboard stopped")
            except:
                self.dashboard_process.kill()
                print("🔥 Dashboard force killed")
    
    def generate_experiment_summary(self):
        """Generate experiment summary"""
        duration = datetime.now() - self.experiment_start_time
        
        print("\n" + "=" * 60)
        print("📊 EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Start Time: {self.experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration}")
        
        # Check for generated data files
        data_dir = Path("results/realtime_data")
        if data_dir.exists():
            data_files = list(data_dir.glob("*.json"))
            print(f"Data Files Generated: {len(data_files)}")
            
            if data_files:
                latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
                print(f"Latest Data File: {latest_file.name}")
        
        print("\n📁 Results Directory: results/")
        print("🌐 Dashboard: http://localhost:8080 (if still running)")
        print("=" * 60)
    
    async def run_full_experiment(self, duration_minutes: int = 30):
        """Run the complete experiment workflow"""
        try:
            # Setup
            self.setup_signal_handlers()
            
            # Start data collection
            if not self.start_data_collector():
                return False
            
            # Start dashboard
            if not self.start_dashboard():
                self.stop_experiment()
                return False
            
            # Open dashboard in browser
            self.open_dashboard()
            
            # Monitor experiment
            self.monitor_experiment(duration_minutes)
            
            # Stop processes
            self.stop_experiment()
            
            # Generate summary
            self.generate_experiment_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Experiment failed: {str(e)}")
            self.stop_experiment()
            return False

def run_quick_demo():
    """Run a quick 5-minute demo"""
    print("🎬 Running Quick Demo (5 minutes)")
    runner = ExperimentRunner()
    asyncio.run(runner.run_full_experiment(duration_minutes=5))

def run_standard_experiment():
    """Run standard 30-minute experiment"""
    print("🔬 Running Standard Experiment (30 minutes)")
    runner = ExperimentRunner()
    asyncio.run(runner.run_full_experiment(duration_minutes=30))

def run_extended_experiment():
    """Run extended 2-hour experiment"""
    print("🧪 Running Extended Experiment (2 hours)")
    runner = ExperimentRunner()
    asyncio.run(runner.run_full_experiment(duration_minutes=120))

def main():
    """Main menu for experiment runner"""
    print("\n🔬 MCP Integrated Experiment Runner")
    print("Select experiment mode:")
    print("1. Quick Demo (5 minutes)")
    print("2. Standard Experiment (30 minutes)")
    print("3. Extended Experiment (2 hours)")
    print("4. Custom Duration")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        run_quick_demo()
    elif choice == "2":
        run_standard_experiment()
    elif choice == "3":
        run_extended_experiment()
    elif choice == "4":
        try:
            duration = int(input("Enter duration in minutes: "))
            if duration > 0:
                print(f"🎯 Running Custom Experiment ({duration} minutes)")
                runner = ExperimentRunner()
                asyncio.run(runner.run_full_experiment(duration_minutes=duration))
            else:
                print("❌ Invalid duration")
        except ValueError:
            print("❌ Invalid input")
    else:
        print("❌ Invalid choice. Running quick demo...")
        run_quick_demo()

if __name__ == "__main__":
    main()