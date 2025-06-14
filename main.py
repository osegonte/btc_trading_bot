#!/usr/bin/env python3
"""
Consolidated BTC Trading Bot - Main Entry Point
6-File Architecture: main.py, data_collection.py, trading_logic.py, trade_execution.py, logger.py, ml_interface.py
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, Optional

# Configuration - Easy to modify
CONFIG = {
    # Trading Configuration
    'symbol': 'BTCUSD',
    'quantity': 0.001,  # 0.001 BTC position size
    'profit_target_ticks': 10,  # $10 profit target
    'stop_loss_ticks': 5,       # $5 stop loss
    'tick_size': 1.0,           # $1 per tick
    'min_confidence': 0.75,     # Minimum signal confidence
    'max_position_time': 30,    # Max 30 seconds in position
    
    # API Configuration (Replace with your keys)
    'paper_trading': True,                          # Set to False for live trading
    'api_key': 'YOUR_ALPACA_API_KEY',              # Replace with your API key
    'secret_key': 'YOUR_ALPACA_SECRET_KEY',        # Replace with your secret key
    
    # Risk Management
    'max_daily_trades': 20,      # Conservative daily limit
    'daily_loss_limit': 500.0,   # Stop trading at $500 daily loss
    'max_position_size': 0.01,   # Maximum position size
    'min_trade_interval': 3.0,   # Minimum 3 seconds between trades
    
    # Bot Settings
    'log_file': 'btcusd_trades.csv',
    'status_update_interval': 10,  # Status updates every 10 seconds
    'weekend_mode': True,          # Enable weekend simulation
    'enable_ml': True,             # Enable ML features
    
    # Performance Monitoring
    'enable_win_rate_monitoring': True,
    'win_rate_alert_threshold': 30.0,    # Alert if win rate drops below 30%
    'export_metrics_interval': 300,      # Export metrics every 5 minutes
}

# Import core modules
from data_collection import EnhancedBTCDataCollector
from trading_logic import EnhancedBTCTradingLogic
from trade_execution import EnhancedBTCTradeExecutor
from logger import SimpleBTCTradeLogger

# Import ML interface
try:
    from ml_interface import BTCMLInterface, BTC_ML_CONFIG
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("⚠️ ML interface not available")


class ConsolidatedBTCBot:
    """Consolidated BTC Trading Bot with all features integrated"""
    
    def __init__(self):
        # Validate configuration
        self._validate_config()
        
        # Bot state
        self.is_running = False
        self.session_start = datetime.now()
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_pnl': 0.0
        }
        
        # Initialize components
        self.setup_logging()
        self.initialize_components()
        
        print(f"\n₿ Consolidated BTC Trading Bot Initialized")
        self.display_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        issues = []
        
        if CONFIG['api_key'] == 'YOUR_ALPACA_API_KEY':
            issues.append("⚠️ API key not configured - will use simulation mode")
        
        if CONFIG['quantity'] > 0.01:
            issues.append("⚠️ Position size might be large for testing")
        
        if CONFIG['min_confidence'] < 0.70:
            issues.append("⚠️ Confidence threshold might be too low")
        
        if issues:
            print("⚠️ Configuration Notes:")
            for issue in issues:
                print(f"   {issue}")
            print()
    
    def setup_logging(self):
        """Setup logging system"""
        log_filename = f'btc_bot_{datetime.now().strftime("%Y%m%d")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        if not CONFIG['paper_trading'] and CONFIG['api_key'] != 'YOUR_ALPACA_API_KEY':
            logging.warning("🔴 LIVE TRADING MODE - REAL MONEY AT RISK!")
    
    def initialize_components(self):
        """Initialize all trading components"""
        
        print("🔧 Initializing BTC trading components...")
        
        # 1. Enhanced Data Collector
        data_config = {
            'simulation': {
                'base_price': 43000.0,
                'volatility_multiplier': 2.0,
                'tick_frequency': 0.1,
                'price_variation': 50.0
            },
            'force_simulation': CONFIG['weekend_mode'] and self._is_weekend()
        }
        self.data_collector = EnhancedBTCDataCollector(CONFIG['symbol'], data_config)
        
        # 2. Enhanced Trading Logic
        logic_config = {
            'profit_target_ticks': CONFIG['profit_target_ticks'],
            'stop_loss_ticks': CONFIG['stop_loss_ticks'],
            'tick_size': CONFIG['tick_size'],
            'min_confidence': CONFIG['min_confidence'],
            'max_position_time': CONFIG['max_position_time'],
            'min_trade_interval': CONFIG['min_trade_interval'],
            'max_daily_loss': CONFIG['daily_loss_limit']
        }
        self.trading_logic = EnhancedBTCTradingLogic(logic_config)
        
        # 3. Enhanced Trade Executor
        execution_config = {
            'paper_trading': CONFIG['paper_trading'],
            'api_key': CONFIG['api_key'] if CONFIG['api_key'] != 'YOUR_ALPACA_API_KEY' else '',
            'secret_key': CONFIG['secret_key'] if CONFIG['secret_key'] != 'YOUR_ALPACA_SECRET_KEY' else '',
            'daily_loss_limit': CONFIG['daily_loss_limit'],
            'max_position_size': CONFIG['max_position_size']
        }
        self.trade_executor = EnhancedBTCTradeExecutor(execution_config)
        
        # 4. Trade Logger
        self.trade_logger = SimpleBTCTradeLogger(CONFIG['log_file'])
        
        # 5. ML Interface (optional)
        if CONFIG['enable_ml'] and ML_AVAILABLE:
            ml_config = BTC_ML_CONFIG.copy()
            ml_config['model_file'] = CONFIG['log_file'].replace('.csv', '_ml_model.pkl')
            self.ml_interface = BTCMLInterface(ml_config)
            print("🤖 ML interface enabled")
        else:
            self.ml_interface = None
            if CONFIG['enable_ml']:
                print("⚠️ ML requested but not available")
        
        # Setup data callback
        self.data_collector.add_tick_callback(self.on_tick_received)
        
        print("✅ All components initialized successfully")
    
    def _is_weekend(self) -> bool:
        """Check if it's weekend"""
        return datetime.now().weekday() >= 5
    
    def display_config(self):
        """Display current configuration"""
        
        # Determine mode
        is_live = not CONFIG['paper_trading'] and CONFIG['api_key'] != 'YOUR_ALPACA_API_KEY'
        mode = "🔴 LIVE" if is_live else "📄 PAPER"
        
        # Determine data source
        if self._is_weekend() and CONFIG['weekend_mode']:
            data_source = "🏖️ WEEKEND SIMULATION"
        elif CONFIG['api_key'] != 'YOUR_ALPACA_API_KEY':
            data_source = "📡 LIVE DATA"
        else:
            data_source = "🎮 SIMULATION"
        
        print(f"   ₿ Symbol: {CONFIG['symbol']}")
        print(f"   💹 Mode: {mode}")
        print(f"   📡 Data: {data_source}")
        print(f"   📦 Position Size: {CONFIG['quantity']} BTC")
        print(f"   🎯 Profit Target: ${CONFIG['profit_target_ticks']}")
        print(f"   🛡️ Stop Loss: ${CONFIG['stop_loss_ticks']}")
        print(f"   📊 Max Daily Trades: {CONFIG['max_daily_trades']}")
        print(f"   💰 Daily Loss Limit: ${CONFIG['daily_loss_limit']}")
        print(f"   🤖 ML Enabled: {'✅' if self.ml_interface else '❌'}")
        
        if is_live:
            print(f"   🚨 WARNING: Real money trading active!")
    
    async def start_trading(self):
        """Start the consolidated trading bot"""
        
        print("\n" + "="*70)
        print("           🚀 STARTING CONSOLIDATED BTC TRADING BOT")
        print("="*70)
        
        try:
            # Start data feed
            print("📡 Starting BTC data feed...")
            api_key = CONFIG['api_key'] if CONFIG['api_key'] != 'YOUR_ALPACA_API_KEY' else ''
            secret_key = CONFIG['secret_key'] if CONFIG['secret_key'] != 'YOUR_ALPACA_SECRET_KEY' else ''
            self.data_collector.start_data_feed(api_key, secret_key)
            
            # Wait for data connection
            await self._wait_for_data()
            
            # Display account info
            self._display_account_info()
            
            # Start main trading loop
            print("🔄 Starting main trading loop...")
            print("⏹️ Press Ctrl+C to stop")
            print("-" * 70)
            
            self.is_running = True
            await self._main_trading_loop()
            
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested...")
            await self._shutdown()
        except Exception as e:
            logging.error(f"Bot error: {e}")
            await self._shutdown()
    
    async def _wait_for_data(self, max_wait: int = 15):
        """Wait for data connection"""
        
        for i in range(max_wait):
            if self.data_collector.tick_count > 0:
                price = self.data_collector.get_current_price()
                data_info = self.data_collector.get_data_source_info()
                print(f"✅ BTC data connected - Price: ${price:,.2f} | Source: {data_info['source']}")
                return True
            
            await asyncio.sleep(1)
            if i % 3 == 0:
                print(f"   Waiting for data... ({i+1}/{max_wait})")
        
        print("⚠️ Data connection timeout - continuing anyway")
        return False
    
    def _display_account_info(self):
        """Display account information"""
        
        account = self.trade_executor.get_account_info()
        print(f"💰 Account Balance: ${account['balance']:,.2f}")
        print(f"💵 Available Cash: ${account['cash']:,.2f}")
        
        if account.get('btc_holdings', 0) > 0:
            print(f"₿ BTC Holdings: {account['btc_holdings']:.6f}")
    
    async def _main_trading_loop(self):
        """Main trading loop with comprehensive monitoring"""
        
        last_status_time = time.time()
        last_metrics_export = time.time()
        
        while self.is_running:
            try:
                # Check circuit breaker
                if self.circuit_breaker_active:
                    print("🔴 Circuit breaker active - waiting...")
                    await asyncio.sleep(60)
                    continue
                
                # Check daily limits
                if self.trades_today >= CONFIG['max_daily_trades']:
                    print(f"🛑 Daily trade limit reached: {self.trades_today}")
                    break
                
                # Check daily loss limit
                if self.daily_pnl <= -CONFIG['daily_loss_limit']:
                    self._activate_circuit_breaker("Daily loss limit exceeded")
                    continue
                
                # Check consecutive losses
                if self.consecutive_losses >= 5:
                    print(f"⚠️ {self.consecutive_losses} consecutive losses - pausing 60s")
                    await asyncio.sleep(60)
                    self.consecutive_losses = 0
                
                # Periodic status update
                current_time = time.time()
                if current_time - last_status_time > CONFIG['status_update_interval']:
                    self._log_status_update()
                    last_status_time = current_time
                
                # Periodic metrics export
                if current_time - last_metrics_export > CONFIG['export_metrics_interval']:
                    self._export_metrics()
                    last_metrics_export = current_time
                
                await asyncio.sleep(0.1)  # Fast loop for responsiveness
                
            except Exception as e:
                logging.error(f"Trading loop error: {e}")
                await asyncio.sleep(1)
        
        await self._shutdown()
    
    def on_tick_received(self, tick_data):
        """Process incoming tick data with comprehensive logic"""
        
        try:
            # Get market analysis
            market_analysis = self.data_collector.get_market_analysis()
            if not market_analysis:
                return
            
            # Get ML signal if available
            ml_signal = None
            if self.ml_interface:
                ml_signal = self.ml_interface.process_tick(tick_data)
            
            # Generate trading signal
            signal = self.trading_logic.evaluate_tick(tick_data, market_analysis, ml_signal)
            
            # Process signals
            if signal.signal_type in ['buy', 'sell']:
                self._execute_signal(signal, tick_data, ml_signal)
            elif signal.signal_type == 'close':
                self._close_position(tick_data, signal.reasoning)
            
        except Exception as e:
            logging.error(f"Tick processing error: {e}")
    
    def _execute_signal(self, signal, tick_data, ml_signal=None):
        """Execute trading signal"""
        
        current_price = tick_data['price']
        
        print(f"\n₿ BTC SIGNAL: {signal.signal_type.upper()} @ ${current_price:,.2f}")
        print(f"   🎯 Confidence: {signal.confidence:.2f}")
        print(f"   💡 Reasoning: {signal.reasoning}")
        
        if ml_signal and ml_signal.signal != 'hold':
            print(f"   🤖 ML: {ml_signal.signal} ({ml_signal.confidence:.2f})")
        
        # Execute trade
        trade = self.trade_executor.place_order(
            CONFIG['symbol'], signal.signal_type, CONFIG['quantity'], current_price
        )
        
        if trade.status.value == "filled":
            # Update trading logic
            self.trading_logic.update_position(
                signal.signal_type, trade.fill_price or current_price, CONFIG['quantity'], trade.timestamp
            )
            
            # Log trade
            self.trade_logger.log_trade(trade, trade_type="entry")
            
            # Update counters
            self.trades_today += 1
            
            print(f"✅ BTC ENTRY: {signal.signal_type.upper()} position opened")
            print(f"   📦 Quantity: {CONFIG['quantity']} BTC")
            print(f"   💰 Value: ${(trade.fill_price or current_price) * CONFIG['quantity']:,.2f}")
            print(f"   🆔 Trade ID: {trade.trade_id}")
            print(f"   📊 Trade #{self.trades_today} today")
            
        else:
            print(f"❌ BTC ENTRY FAILED: {trade.status.value}")
    
    def _close_position(self, tick_data, reasoning):
        """Close current position"""
        
        current_price = tick_data['price']
        position_info = self.trading_logic.get_position_info()
        
        if not position_info['has_position']:
            return
        
        print(f"\n₿ BTC EXIT: {reasoning} @ ${current_price:,.2f}")
        
        # Close via trade executor
        exit_trade = self.trade_executor.close_position(current_price)
        
        if exit_trade and exit_trade.status.value == "filled":
            # Calculate P&L
            entry_price = position_info['entry_price']
            if position_info['side'] == 'long':
                pnl = (current_price - entry_price) * CONFIG['quantity']
            else:
                pnl = (entry_price - current_price) * CONFIG['quantity']
            
            # Update performance tracking
            self._update_performance_metrics(pnl)
            
            # Update trading logic
            self.trading_logic.update_position('close', current_price, CONFIG['quantity'])
            
            # Log exit trade
            self.trade_logger.log_trade(exit_trade, trade_type="exit", profit_loss=pnl)
            
            # Record ML outcome
            if self.ml_interface:
                features = self.ml_interface.feature_extractor.extract_features()
                if features:
                    self.ml_interface.record_trade_outcome(features, exit_trade.side, pnl)
            
            # Display results
            pnl_symbol = "🟢" if pnl > 0 else "🔴"
            pnl_percentage = (pnl / (entry_price * CONFIG['quantity'])) * 100 if entry_price > 0 else 0
            
            print(f"✅ BTC EXIT: Position closed")
            print(f"   💰 P&L: ${pnl:+.2f} ({pnl_percentage:+.2f}%) {pnl_symbol}")
            print(f"   🆔 Trade ID: {exit_trade.trade_id}")
            
        else:
            print(f"❌ BTC EXIT FAILED")
    
    def _update_performance_metrics(self, pnl: float):
        """Update comprehensive performance metrics"""
        
        self.performance_metrics['total_trades'] += 1
        self.performance_metrics['total_pnl'] += pnl
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.performance_metrics['winning_trades'] += 1
            self.consecutive_losses = 0
        else:
            self.performance_metrics['losing_trades'] += 1
            self.consecutive_losses += 1
        
        # Calculate win rate
        total_trades = self.performance_metrics['total_trades']
        if total_trades > 0:
            self.performance_metrics['win_rate'] = (self.performance_metrics['winning_trades'] / total_trades) * 100
            self.performance_metrics['avg_pnl'] = self.performance_metrics['total_pnl'] / total_trades
        
        # Update peak and drawdown
        if self.performance_metrics['total_pnl'] > self.performance_metrics['peak_pnl']:
            self.performance_metrics['peak_pnl'] = self.performance_metrics['total_pnl']
        
        current_drawdown = self.performance_metrics['peak_pnl'] - self.performance_metrics['total_pnl']
        if current_drawdown > self.performance_metrics['max_drawdown']:
            self.performance_metrics['max_drawdown'] = current_drawdown
        
        # Win rate monitoring
        if (CONFIG['enable_win_rate_monitoring'] and 
            total_trades >= 10 and 
            self.performance_metrics['win_rate'] < CONFIG['win_rate_alert_threshold']):
            logging.warning(f"⚠️ WIN RATE ALERT: {self.performance_metrics['win_rate']:.1f}% below threshold")
    
    def _activate_circuit_breaker(self, reason: str):
        """Activate circuit breaker"""
        self.circuit_breaker_active = True
        print(f"\n🔴 CIRCUIT BREAKER ACTIVATED: {reason}")
        print(f"🔴 Daily P&L: ${self.daily_pnl:.2f}")
        logging.critical(f"CIRCUIT BREAKER: {reason}")
        
        # Auto-deactivate after 1 hour
        import threading
        def deactivate():
            time.sleep(3600)
            self.circuit_breaker_active = False
            print("🟢 Circuit breaker deactivated")
        
        threading.Thread(target=deactivate, daemon=True).start()
    
    def _log_status_update(self):
        """Log comprehensive status update"""
        
        current_time = datetime.now().strftime('%H:%M:%S')
        current_price = self.data_collector.get_current_price()
        data_info = self.data_collector.get_data_source_info()
        position_info = self.trading_logic.get_position_info()
        
        print(f"\n₿ BTC STATUS - {current_time}")
        print(f"   💹 {CONFIG['symbol']}: ${current_price:,.2f}")
        print(f"   📡 Source: {data_info['source']} | Ticks: {data_info['tick_count']}")
        
        # Position information
        if position_info['has_position']:
            unrealized_pnl = self.trading_logic.position.unrealized_pnl
            time_in_pos = position_info['time_in_position']
            print(f"   📍 Position: {position_info['side'].upper()} @ ${position_info['entry_price']:,.2f}")
            print(f"   💰 Unrealized: ${unrealized_pnl:+.2f} ({time_in_pos:.0f}s)")
        else:
            print(f"   📍 Position: NONE")
        
        # Trading metrics
        print(f"   📊 Trades Today: {self.trades_today}/{CONFIG['max_daily_trades']}")
        print(f"   🎯 Win Rate: {self.performance_metrics['win_rate']:.1f}%")
        print(f"   💰 Daily P&L: ${self.daily_pnl:+.2f}")
        
        # Risk metrics
        risk_utilization = (abs(self.daily_pnl) / CONFIG['daily_loss_limit']) * 100 if self.daily_pnl < 0 else 0
        print(f"   ⚠️ Risk Level: {risk_utilization:.1f}%")
        
        # ML stats
        if self.ml_interface:
            ml_stats = self.ml_interface.get_ml_stats()
            if ml_stats.get('model_trained', False):
                print(f"   🤖 ML: {ml_stats['accuracy']:.1f}% accuracy (v{ml_stats.get('model_version', 1)})")
        
        # Connection health
        health = self.data_collector.get_connection_health()
        if health['health_score'] < 70:
            print(f"   ⚠️ Connection: {health['status']} ({health['health_score']}/100)")
    
    def _export_metrics(self):
        """Export performance metrics to file"""
        
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'session_duration_minutes': session_duration,
            'trades_completed': self.performance_metrics['total_trades'],
            'win_rate': self.performance_metrics['win_rate'],
            'total_pnl': self.performance_metrics['total_pnl'],
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.performance_metrics['max_drawdown'],
            'consecutive_losses': self.consecutive_losses,
            'circuit_breaker_active': self.circuit_breaker_active,
            'data_source': self.data_collector.get_data_source_info()['source']
        }
        
        # Add ML metrics if available
        if self.ml_interface:
            ml_stats = self.ml_interface.get_ml_stats()
            metrics['ml_accuracy'] = ml_stats.get('accuracy', 0)
            metrics['ml_model_version'] = ml_stats.get('model_version', 0)
        
        # Save to file
        metrics_file = CONFIG['log_file'].replace('.csv', '_metrics.json')
        try:
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
        except Exception as e:
            logging.error(f"Failed to export metrics: {e}")
    
    async def _shutdown(self):
        """Comprehensive shutdown procedure"""
        
        print("\n🛑 Shutting down Consolidated BTC Bot...")
        self.is_running = False
        
        try:
            # Close any open positions
            position_info = self.trading_logic.get_position_info()
            if position_info['has_position']:
                print("🔄 Closing open position...")
                current_price = self.data_collector.get_current_price()
                self._close_position({'price': current_price}, "Bot shutdown")
            
            # Stop data feed
            self.data_collector.stop_data_feed()
            
            # Final ML training
            if self.ml_interface:
                print("🤖 Final ML model training...")
                self.ml_interface.force_retrain()
            
            # Export final metrics
            self._export_metrics()
            
            # Generate final report
            self._generate_final_report()
            
            # Cleanup
            self.trade_logger.cleanup()
            
            print("✅ Consolidated BTC Bot shutdown completed")
            
        except Exception as e:
            logging.error(f"Shutdown error: {e}")
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        
        session_duration = datetime.now() - self.session_start
        
        print("\n" + "="*80)
        print("           ₿ CONSOLIDATED BTC TRADING FINAL REPORT")
        print("="*80)
        
        # Session overview
        print(f"Session Duration: {str(session_duration).split('.')[0]}")
        print(f"Trading Mode: {'🔴 LIVE' if not CONFIG['paper_trading'] else '📄 PAPER'}")
        print(f"Data Source: {self.data_collector.get_data_source_info()['source'].upper()}")
        
        # Performance metrics
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"Total Trades: {self.performance_metrics['total_trades']}")
        print(f"Win Rate: {self.performance_metrics['win_rate']:.1f}%")
        print(f"Total P&L: ${self.performance_metrics['total_pnl']:+.2f}")
        print(f"Daily P&L: ${self.daily_pnl:+.2f}")
        print(f"Average P&L: ${self.performance_metrics['avg_pnl']:+.2f}")
        print(f"Max Drawdown: ${self.performance_metrics['max_drawdown']:.2f}")
        
        # Risk analysis
        risk_utilization = (abs(self.daily_pnl) / CONFIG['daily_loss_limit']) * 100 if self.daily_pnl < 0 else 0
        print(f"\n⚠️ RISK ANALYSIS:")
        print(f"Risk Utilization: {risk_utilization:.1f}% of daily limit")
        print(f"Circuit Breaker Triggered: {'Yes' if self.circuit_breaker_active else 'No'}")
        print(f"Max Consecutive Losses: {self.consecutive_losses}")
        
        # Trading efficiency
        trades_per_hour = self.performance_metrics['total_trades'] / max(1, session_duration.total_seconds() / 3600)
        print(f"\n📈 EFFICIENCY METRICS:")
        print(f"Trading Frequency: {trades_per_hour:.1f} trades/hour")
        print(f"Position Utilization: {self.trades_today}/{CONFIG['max_daily_trades']} daily limit")
        
        # ML performance
        if self.ml_interface:
            ml_stats = self.ml_interface.get_ml_stats()
            print(f"\n🤖 ML PERFORMANCE:")
            print(f"Model Version: {ml_stats.get('model_version', 'N/A')}")
            print(f"Accuracy: {ml_stats.get('accuracy', 0):.1f}%")
            print(f"Training Samples: {ml_stats.get('training_samples', 0)}")
        
        # Trade executor stats
        exec_stats = self.trade_executor.get_trading_stats()
        print(f"\n📋 EXECUTION STATS:")
        print(f"Order Success Rate: {exec_stats['success_rate']:.1f}%")
        print(f"Average Slippage: ${exec_stats['avg_slippage']:.2f}")
        print(f"Total Commission: ${exec_stats['total_commission']:.2f}")
        
        print("="*80)


def validate_setup():
    """Validate setup before starting"""
    print("🔍 Validating setup...")
    
    # Check if we're in live mode
    is_live = not CONFIG['paper_trading'] and CONFIG['api_key'] != 'YOUR_ALPACA_API_KEY'
    
    if is_live:
        print("🔴 LIVE TRADING MODE DETECTED!")
        print("🔴 This will trade with REAL MONEY!")
        print("🔴 Make sure you understand the risks!")
        
        # Give user time to abort
        for i in range(5, 0, -1):
            print(f"🔴 Starting in {i} seconds... (Press Ctrl+C to abort)")
            time.sleep(1)
    
    print("✅ Setup validation completed")


async def main():
    """Main entry point"""
    
    print("₿ Consolidated BTC Trading Bot")
    print("=" * 50)
    print("6-File Architecture: Enhanced & Streamlined")
    print()
    
    # Validate setup
    validate_setup()
    
    # Create and start bot
    bot = ConsolidatedBTCBot()
    await bot.start_trading()


if __name__ == "__main__":
    asyncio.run(main())