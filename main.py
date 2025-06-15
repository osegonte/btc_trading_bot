#!/usr/bin/env python3
"""
Complete BTC Swing Trading Bot - €20 to €1M Challenge
Main integration file for swing trading system with enhanced ML learning
Key Features: 2-5 minute swing positions, market structure awareness, sustainable growth
OPTIMIZED: Lowered confidence thresholds for more active trading
DEBUG VERSION: Enhanced debugging and nuclear logic test
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict

# Import the 5 core swing trading modules
from data_collection import BTCSwingDataCollector
from trading_logic import BTCSwingLogic, SwingSignalType
from trade_execution import BTCSwingExecutor
from logger import BTCSwingLogger
from ml_interface import BTCSwingMLInterface, BTC_SWING_ML_CONFIG

# Configuration for €20 to €1M Swing Challenge - OPTIMIZED
SWING_CONFIG = {
    # Challenge Parameters
    'starting_balance': 20.0,                    # Start with €20
    'profit_target_pct': 2.5,                   # 2.5% profit target
    'stop_loss_pct': 1.0,                       # 1.0% stop loss
    'max_position_time': 300,                   # 5 minutes max hold
    'min_position_time': 120,                   # 2 minutes min hold
    'min_confidence': 0.45,                     # OPTIMIZED: Reduced from 0.65 to 0.55
    'risk_per_trade_pct': 1.5,                  # 1.5% risk per trade
    'position_multiplier': 1.5,                 # 1.5x sustainable multiplier
    
    # API Configuration
    'paper_trading': True,
    'api_key': 'YOUR_ALPACA_API_KEY',           # Replace with your keys
    'secret_key': 'YOUR_ALPACA_SECRET_KEY',     # Replace with your keys
    
    # Swing Trading Settings
    'max_daily_trades': 25,                     # Reasonable for swing trading
    'signal_cooldown': 10,                      # OPTIMIZED: Reduced from 30 to 20 seconds
    'status_update_interval': 15,               # Status every 15 seconds
    
    # ML Learning Configuration
    'ml_enabled': True,                         # Enable machine learning
    'ml_min_confidence': 0.40,                  # OPTIMIZED: Reduced from 0.60 to 0.50
    'auto_ml_retrain': True,                    # Automatically retrain model
    
    # Risk Management
    'max_consecutive_losses': 3,                # Reset after 3 losses
    'daily_loss_limit_pct': 10.0,              # 10% daily loss limit
    'force_reset_balance': 5.0,                 # Reset if below €5
    
    # Files
    'log_file': 'btc_swing_challenge.csv',
    'ml_model_file': 'btc_swing_ml_model.pkl'
}


class BTCSwingTradingBot:
    """
    Complete BTC Swing Trading Bot for €20 to €1M Challenge
    Enhanced for sustainable swing trading with 2-5 minute positions
    OPTIMIZED: More responsive signal generation
    DEBUG VERSION: Enhanced debugging capabilities
    """
    
    def __init__(self):
        # Bot state
        self.is_running = False
        self.session_start = datetime.now()
        self.current_balance = SWING_CONFIG['starting_balance']
        self.trades_today = 0
        self.consecutive_losses = 0
        self.challenge_attempt = 1
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_pnl = 0.0
        self.total_hold_time = 0
        
        # ML Learning state
        self.ml_enabled = SWING_CONFIG['ml_enabled']
        self.ml_predictions_today = 0
        self.ml_correct_today = 0
        self.last_ml_retrain = None
        
        # Initialize components
        self.setup_logging()
        self.initialize_components()
        
        print(f"\n₿ BTC SWING TRADING BOT v3.0 - OPTIMIZED - DEBUG VERSION")
        print(f"🎯 Challenge: €20 → €1,000,000")
        print(f"💰 Starting Balance: €{self.current_balance}")
        print(f"📊 Strategy: {SWING_CONFIG['profit_target_pct']}% target, {SWING_CONFIG['stop_loss_pct']}% stop")
        print(f"⏱️ Hold Time: {SWING_CONFIG['min_position_time']}-{SWING_CONFIG['max_position_time']} seconds")
        print(f"🤖 ML Learning: {'ENABLED' if self.ml_enabled else 'DISABLED'}")
        print(f"🔄 Position Multiplier: {SWING_CONFIG['position_multiplier']}x")
        print(f"🎯 OPTIMIZED: Confidence {SWING_CONFIG['min_confidence']} (was 0.65)")
        print(f"🐛 DEBUG MODE: Enhanced signal debugging enabled")
    
    def setup_logging(self):
        """Setup logging for the swing trading bot"""
        log_filename = f'btc_swing_{datetime.now().strftime("%Y%m%d")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        logging.info("🚀 BTC Swing Trading Bot v3.0 starting... OPTIMIZED - DEBUG VERSION")
    
    def initialize_components(self):
        """Initialize all swing trading components"""
        
        print("🔧 Initializing optimized swing trading components...")
        
        # 1. Data Collection (Swing-focused)
        self.data_collector = BTCSwingDataCollector("BTCUSD")
        
        # 2. Trading Logic (Swing configuration) - OPTIMIZED
        swing_logic_config = {
            'profit_target_pct': SWING_CONFIG['profit_target_pct'],
            'stop_loss_pct': SWING_CONFIG['stop_loss_pct'],
            'min_confidence': SWING_CONFIG['min_confidence'],          # Now 0.55
            'max_position_time': SWING_CONFIG['max_position_time'],
            'min_position_time': SWING_CONFIG['min_position_time'],
            'risk_per_trade_pct': SWING_CONFIG['risk_per_trade_pct'],
            'position_multiplier': SWING_CONFIG['position_multiplier'],
            'signal_cooldown': SWING_CONFIG['signal_cooldown']        # Now 20 seconds
        }
        self.trading_logic = BTCSwingLogic(swing_logic_config)
        self.trading_logic.current_balance = self.current_balance
        
        # 3. Trade Execution (Swing-optimized)
        execution_config = {
            'paper_trading': SWING_CONFIG['paper_trading'],
            'api_key': SWING_CONFIG['api_key'] if SWING_CONFIG['api_key'] != 'YOUR_ALPACA_API_KEY' else '',
            'secret_key': SWING_CONFIG['secret_key'] if SWING_CONFIG['secret_key'] != 'YOUR_ALPACA_SECRET_KEY' else ''
        }
        self.trade_executor = BTCSwingExecutor(execution_config)
        
        # 4. Trade Logger (Swing-enhanced)
        self.trade_logger = BTCSwingLogger(SWING_CONFIG['log_file'])
        
        # 5. ML Interface (Swing-optimized) - OPTIMIZED
        if self.ml_enabled:
            ml_config = BTC_SWING_ML_CONFIG.copy()
            ml_config['model_file'] = SWING_CONFIG['ml_model_file']
            ml_config['min_confidence'] = SWING_CONFIG['ml_min_confidence']  # Now 0.50
            self.ml_interface = BTCSwingMLInterface(ml_config)
            
            # Connect ML to trading logic
            self.trading_logic.set_ml_interface(self.ml_interface)
            
            print("🤖 ML Learning System initialized for swing trading")
            print(f"   Model file: {SWING_CONFIG['ml_model_file']}")
            print(f"   Min confidence: {SWING_CONFIG['ml_min_confidence']} (OPTIMIZED)")
        else:
            self.ml_interface = None
            print("🚫 ML Learning disabled")
        
        # Connect data collector to trading logic via candle callbacks
        self.data_collector.add_candle_callback(self.on_candle_completed)
        
        print("✅ All optimized swing trading components initialized")
    
    async def start_swing_trading(self):
        """Start the BTC swing trading bot"""
        
        print("\n" + "="*80)
        print("      🚀 STARTING €20 → €1M BTC SWING TRADING CHALLENGE v3.0 - OPTIMIZED - DEBUG")
        print("="*80)
        
        try:
            # Start data feed
            print("📡 Starting BTC swing data feed...")
            api_key = SWING_CONFIG['api_key'] if SWING_CONFIG['api_key'] != 'YOUR_ALPACA_API_KEY' else ''
            secret_key = SWING_CONFIG['secret_key'] if SWING_CONFIG['secret_key'] != 'YOUR_ALPACA_SECRET_KEY' else ''
            self.data_collector.start_data_feed(api_key, secret_key)
            
            # Wait for data connection
            await self._wait_for_data()
            
            # Display initial account info
            self._display_account_info()
            
            # Start main swing trading loop
            print("🔄 Starting optimized swing trading loop...")
            print("💡 Strategy: 2.5% profits on 2-5 minute swing positions")
            print("🤖 ML: Learning from swing trade patterns and market structure")
            print("🔄 Sustainable: 1.5x position multiplier for consistent growth")
            print("🎯 OPTIMIZED: Lower confidence thresholds for more active trading")
            print("🐛 DEBUG: Enhanced signal analysis and logging")
            print("⏹️ Press Ctrl+C to stop")
            print("-" * 80)
            
            self.is_running = True
            await self._main_swing_trading_loop()
            
        except KeyboardInterrupt:
            print("\n🛑 Swing trading session stopped by user")
            await self._shutdown()
        except Exception as e:
            logging.error(f"Bot error: {e}")
            print(f"❌ Bot error: {e}")
            await self._shutdown()
    
    async def _wait_for_data(self, max_wait: int = 10):
        """Wait for data connection"""
        
        for i in range(max_wait):
            status = self.data_collector.get_connection_status()
            if status['tick_count'] > 0:
                print(f"✅ BTC data connected - Price: €{status['current_price']:,.2f} | Source: {status['source']}")
                return True
            
            await asyncio.sleep(1)
            if i % 3 == 0:
                print(f"   Waiting for data... ({i+1}/{max_wait})")
        
        print("⚠️ Data connection timeout - continuing anyway")
        return False
    
    def _display_account_info(self):
        """Display account information"""
        
        account = self.trade_executor.get_account_info()
        print(f"💰 Account Balance: €{account['balance']:,.2f}")
        print(f"💵 Available Cash: €{account['cash']:,.2f}")
        print(f"₿ BTC Holdings: {account['btc_holdings']:.6f}")
        print(f"🔗 Connection: {account['connection_type']}")
        print(f"🔄 Swing Trading Mode: {account.get('swing_trading_mode', True)}")
        
        # Show short selling capability
        if account.get('short_selling_enabled', False):
            print(f"📈📉 Short Selling: ENABLED")
        
        # ML status
        if self.ml_interface:
            ml_stats = self.ml_interface.get_ml_stats()
            print(f"🤖 ML Model: v{ml_stats.get('model_version', 1)} | Samples: {ml_stats.get('training_samples', 0)}")
            print(f"🎯 ML Threshold: {SWING_CONFIG['ml_min_confidence']} (OPTIMIZED)")
            if ml_stats.get('accuracy', 0) > 0:
                print(f"🎯 ML Accuracy: {ml_stats['accuracy']:.1f}%")
    
    async def _main_swing_trading_loop(self):
        """Main swing trading loop with enhanced monitoring"""
        
        last_status_time = time.time()
        last_balance_check = time.time()
        last_ml_update = time.time()
        
        while self.is_running:
            try:
                # Check if balance needs reset
                if time.time() - last_balance_check > 60:  # Check every minute
                    if self._should_reset_challenge():
                        self._reset_challenge()
                    last_balance_check = time.time()
                
                # Check daily limits
                if self.trades_today >= SWING_CONFIG['max_daily_trades']:
                    print(f"🛑 Daily trade limit reached: {self.trades_today}")
                    break
                
                # Check consecutive losses
                if self.consecutive_losses >= SWING_CONFIG['max_consecutive_losses']:
                    print(f"⚠️ {self.consecutive_losses} consecutive losses - pausing 60s")
                    await asyncio.sleep(60)
                    self.consecutive_losses = 0
                
                # ML learning updates
                if self.ml_enabled and time.time() - last_ml_update > 120:  # Every 2 minutes
                    self._update_ml_learning()
                    last_ml_update = time.time()
                
                # Periodic status update
                current_time = time.time()
                if current_time - last_status_time > SWING_CONFIG['status_update_interval']:
                    self._display_status()
                    last_status_time = current_time
                
                await asyncio.sleep(1.0)  # Slower loop for swing trading
                
            except Exception as e:
                logging.error(f"Swing trading loop error: {e}")
                await asyncio.sleep(5)
        
        await self._shutdown()
    
    def on_candle_completed(self, candle_data):
        """Process completed candle for swing trading opportunities - DEBUG VERSION"""
        
        try:
            # DEBUG: Show that candles are being processed
            print(f"🕯️ DEBUG: {candle_data['timeframe']} candle @ €{candle_data['close']:.2f} | Vol: {candle_data['volume']:.2f}")
            
            # Get swing metrics for enhanced analysis
            swing_metrics = self.data_collector.get_swing_metrics()
            
            if swing_metrics.get('insufficient_data'):
                print(f"🔍 DEBUG: Insufficient data - need more candles")
                return
            
            print(f"🔍 DEBUG: Swing metrics available - calling trading logic")
            
            # Update trading logic balance
            self.trading_logic.current_balance = self.current_balance
            
            # Generate swing trading signal
            signal = self.trading_logic.evaluate_candle(candle_data, swing_metrics)
            
            # DEBUG: ALWAYS show signal analysis
            print(f"🔍 DEBUG Signal: {signal.signal_type.value} | Confidence: {signal.confidence:.2f} | {signal.reasoning}")
            
            # Process signals
            if signal.signal_type in [SwingSignalType.BUY, SwingSignalType.SELL]:
                print(f"🚀 DEBUG: EXECUTING SIGNAL!")
                self._execute_swing_signal(signal, candle_data)
            elif signal.signal_type == SwingSignalType.CLOSE:
                print(f"🚀 DEBUG: CLOSING POSITION!")
                self._close_swing_position(candle_data, signal.reasoning)
            else:
                print(f"💤 DEBUG: Holding - {signal.reasoning}")
            
        except Exception as e:
            print(f"❌ DEBUG: Candle processing error: {e}")
            logging.error(f"Candle processing error: {e}")
    
    def _execute_swing_signal(self, signal, candle_data):
        """Execute swing trading signal with enhanced position sizing"""
        
        current_price = candle_data['close']
        
        # Calculate swing position size
        position_size = self.trade_executor.calculate_swing_position_size(
            current_price, 
            self.current_balance,
            SWING_CONFIG['stop_loss_pct'],
            SWING_CONFIG['position_multiplier']
        )
        position_value = position_size * current_price
        
        print(f"\n₿ SWING SIGNAL: {signal.signal_type.value.upper()} @ €{current_price:,.2f}")
        print(f"   🎯 Confidence: {signal.confidence:.2f} (threshold: {SWING_CONFIG['min_confidence']})")
        print(f"   💡 Reasoning: {signal.reasoning}")
        print(f"   📊 Position: {position_size:.6f} BTC (€{position_value:.2f})")
        print(f"   ⏱️ Expected Hold: {signal.expected_hold_time//60}m{signal.expected_hold_time%60}s")
        
        # ML enhancement information
        if self.ml_enabled and 'ML' in signal.reasoning:
            print(f"   🤖 ML Enhanced Signal")
            self.ml_predictions_today += 1
        
        # Execute swing trade
        order = self.trade_executor.place_order(
            "BTCUSD", signal.signal_type.value, position_size, current_price
        )
        
        if order.status.value == "filled":
            # Update trading logic
            self.trading_logic.update_position(
                signal.signal_type.value, 
                order.fill_price or current_price, 
                position_size, 
                order.timestamp
            )
            
            # Log entry trade
            self.trade_logger.log_trade(order, 'entry', current_balance=self.current_balance)
            
            # Update counters
            self.trades_today += 1
            
            direction = "📈 LONG" if signal.signal_type.value == 'buy' else "📉 SHORT"
            print(f"✅ SWING ENTRY: {direction} position opened")
            print(f"   🎯 Target: €{signal.target_price:,.2f} (+{SWING_CONFIG['profit_target_pct']}%)")
            print(f"   🛡️ Stop: €{signal.stop_price:,.2f} (-{SWING_CONFIG['stop_loss_pct']}%)")
            print(f"   📊 Trade #{self.trades_today} today")
            
        else:
            print(f"❌ SWING ENTRY FAILED: {order.status.value}")
    
    def _close_swing_position(self, candle_data, reasoning):
        """Close current swing position with ML learning"""
        
        current_price = candle_data['close']
        position_info = self.trading_logic.get_position_info()
        
        if not position_info['has_position']:
            return
        
        print(f"\n₿ SWING EXIT: {reasoning} @ €{current_price:,.2f}")
        
        # Close position
        exit_order = self.trade_executor.close_position(
            "BTCUSD", 
            position_info['quantity'], 
            current_price
        )
        
        if exit_order and exit_order.status.value == "filled":
            # Calculate P&L
            entry_price = position_info['entry_price']
            quantity = position_info['quantity']
            
            if position_info['side'] == 'long':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Calculate actual euro P&L
            pnl_euros = (pnl_pct / 100) * self.current_balance * (SWING_CONFIG['risk_per_trade_pct'] / 100) * SWING_CONFIG['position_multiplier'] / (SWING_CONFIG['stop_loss_pct'] / 100)
            
            # Account for commissions
            pnl_euros -= (exit_order.commission + getattr(exit_order, 'entry_commission', 0))
            
            # Calculate hold time
            if hasattr(position_info, 'entry_time'):
                hold_time = (datetime.now() - datetime.fromisoformat(position_info['entry_time'])).total_seconds()
            else:
                hold_time = 180  # Default estimate
            
            # Update balance and performance
            self.current_balance += pnl_euros
            self.daily_pnl += pnl_euros
            self.total_trades += 1
            self.total_hold_time += hold_time
            
            # Track ML accuracy and wins/losses
            if self.ml_enabled and self.ml_predictions_today > 0:
                if pnl_euros > 0:
                    self.ml_correct_today += 1
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
            else:
                if pnl_euros > 0:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
            
            # Update trading logic (this triggers ML learning)
            self.trading_logic.update_position('close', current_price, quantity)
            self.trading_logic.current_balance = self.current_balance
            
            # Log exit trade with swing-specific metrics
            self.trade_logger.log_trade(
                exit_order, 'exit', 
                profit_loss=pnl_euros, 
                profit_loss_pct=pnl_pct,
                hold_time=int(hold_time),
                current_balance=self.current_balance
            )
            
            # Display swing results
            pnl_symbol = "🟢" if pnl_euros > 0 else "🔴"
            direction = "📈 LONG" if position_info['side'] == 'long' else "📉 SHORT"
            win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
            avg_hold = (self.total_hold_time / self.total_trades) / 60 if self.total_trades > 0 else 0
            ml_accuracy = (self.ml_correct_today / self.ml_predictions_today) * 100 if self.ml_predictions_today > 0 else 0
            
            print(f"✅ SWING CLOSED: {direction} position")
            print(f"   💰 P&L: €{pnl_euros:+.2f} ({pnl_pct:+.2f}%) {pnl_symbol}")
            print(f"   ⏱️ Hold Time: {int(hold_time//60)}m{int(hold_time%60)}s")
            print(f"   💵 Balance: €{self.current_balance:.2f}")
            print(f"   📊 Session: {self.winning_trades}W/{self.total_trades - self.winning_trades}L ({win_rate:.1f}%) | Avg Hold: {avg_hold:.1f}m")
            if self.ml_enabled and self.ml_predictions_today > 0:
                print(f"   🤖 ML Today: {self.ml_correct_today}/{self.ml_predictions_today} ({ml_accuracy:.1f}%)")
            
            # Check if reached next level
            self._check_level_progress()
            
        else:
            print(f"❌ SWING EXIT FAILED")
    
    def _update_ml_learning(self):
        """Update ML learning system for swing trading"""
        if not self.ml_interface:
            return
        
        try:
            # Check if model needs retraining
            ml_stats = self.ml_interface.get_ml_stats()
            
            if (SWING_CONFIG['auto_ml_retrain'] and 
                ml_stats.get('training_samples', 0) >= 15 and  # OPTIMIZED: Reduced from 20 to 15
                ml_stats.get('training_samples', 0) % 15 == 0):  # OPTIMIZED: Train more frequently
                
                self.ml_interface.force_retrain()
                self.last_ml_retrain = datetime.now()
                print(f"\n🤖 SWING ML MODEL RETRAINED!")
                print(f"   📊 Samples: {ml_stats.get('training_samples', 0)}")
                logging.info("🤖 Swing ML model auto-retrained")
                
        except Exception as e:
            logging.warning(f"ML update error: {e}")
    
    def _check_level_progress(self):
        """Check if reached next challenge level"""
        
        current_level = 0
        target = 20.0
        
        # Calculate current level
        while target <= self.current_balance and target < 1000000:
            current_level += 1
            target *= 2
        
        next_target = min(target, 1000000)
        
        # Check for level completion
        if self.current_balance >= next_target and next_target <= 1000000:
            print(f"\n🎉 SWING LEVEL {current_level + 1} REACHED! €{self.current_balance:.2f}")
            logging.info(f"🎉 Swing Challenge Level {current_level + 1} reached: €{self.current_balance:.2f}")
            
            # ML celebration
            if self.ml_enabled:
                ml_accuracy = (self.ml_correct_today / max(1, self.ml_predictions_today)) * 100
                print(f"🤖 Swing ML contributed with {ml_accuracy:.1f}% accuracy")
            
            if next_target >= 1000000:
                print("🏆 SWING CHALLENGE COMPLETED! €1,000,000 REACHED!")
                logging.info("🏆 Swing €20 to €1M Challenge COMPLETED!")
                self.is_running = False
    
    def _should_reset_challenge(self) -> bool:
        """Check if challenge should be reset"""
        return self.current_balance < SWING_CONFIG['force_reset_balance']
    
    def _reset_challenge(self):
        """Reset challenge to €20 but keep ML knowledge"""
        
        print(f"\n🔄 SWING CHALLENGE RESET (ML KNOWLEDGE RETAINED)")
        print(f"   Previous balance: €{self.current_balance:.2f}")
        print(f"   Trades completed: {self.total_trades}")
        print(f"   Average hold time: {(self.total_hold_time / max(1, self.total_trades)) / 60:.1f} minutes")
        if self.ml_enabled:
            ml_accuracy = (self.ml_correct_today / max(1, self.ml_predictions_today)) * 100
            print(f"   ML accuracy this session: {ml_accuracy:.1f}%")
        
        # Start new attempt
        self.challenge_attempt += 1
        self.trade_logger.start_new_challenge_attempt()
        
        # Reset trading metrics but keep ML learning
        self.current_balance = SWING_CONFIG['starting_balance']
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.total_hold_time = 0
        
        # Reset daily ML counters but keep model knowledge
        self.ml_predictions_today = 0
        self.ml_correct_today = 0
        
        # Reset trading logic
        self.trading_logic.reset_to_twenty_euros()
        self.trading_logic.current_balance = self.current_balance
        
        print(f"✅ Swing challenge attempt #{self.challenge_attempt} started with €20")
        print("🤖 Swing ML model retained previous learning")
        logging.info(f"Swing Challenge reset - Attempt #{self.challenge_attempt} started (ML retained)")
    
    def _display_status(self):
        """Display current swing trading status with enhanced metrics"""
        
        current_price = self.data_collector.get_current_price()
        position_info = self.trading_logic.get_position_info()
        swing_metrics = self.data_collector.get_swing_metrics()
        
        # Calculate level and progress
        current_level = 0
        target = 20.0
        while target <= self.current_balance and target < 1000000:
            current_level += 1
            target *= 2
        next_target = min(target, 1000000)
        progress = (self.current_balance / next_target) * 100
        
        print(f"\n₿ SWING TRADING STATUS - {datetime.now().strftime('%H:%M:%S')} - OPTIMIZED - DEBUG")
        print(f"   💹 BTC Price: €{current_price:,.2f}")
        print(f"   💰 Balance: €{self.current_balance:.2f}")
        print(f"   📊 Level: {current_level} → {current_level + 1} ({progress:.1f}%)")
        print(f"   🎯 Next Target: €{next_target:,.0f}")
        print(f"   📈 Trend: {swing_metrics.get('trend_direction', 'unknown')}")
        print(f"   🎯 Confidence: {SWING_CONFIG['min_confidence']} (OPTIMIZED)")
        
        if position_info['has_position']:
            time_in_pos = position_info['time_in_position']
            direction = "📈 LONG" if position_info['side'] == 'long' else "📉 SHORT"
            current_pnl = position_info.get('current_pnl_pct', 0)
            print(f"   📍 Position: {direction} @ €{position_info['entry_price']:,.2f} ({time_in_pos:.0f}s) | P&L: {current_pnl:+.2f}%")
        else:
            print(f"   📍 Position: NONE")
        
        # Performance metrics
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        avg_hold = (self.total_hold_time / max(1, self.total_trades)) / 60  # minutes
        print(f"   📈 Trades: {self.trades_today}/{SWING_CONFIG['max_daily_trades']} today")
        print(f"   🏆 Record: {self.winning_trades}W/{self.total_trades - self.winning_trades}L ({win_rate:.1f}%)")
        print(f"   💵 Daily P&L: €{self.daily_pnl:+.2f}")
        print(f"   ⏱️ Avg Hold: {avg_hold:.1f} minutes")
        
        # ML Learning metrics
        if self.ml_enabled:
            ml_accuracy = (self.ml_correct_today / max(1, self.ml_predictions_today)) * 100
            ml_stats = self.ml_interface.get_ml_stats() if self.ml_interface else {}
            print(f"   🤖 ML Today: {self.ml_correct_today}/{self.ml_predictions_today} ({ml_accuracy:.1f}%)")
            print(f"   🧠 Model: v{ml_stats.get('model_version', 1)} | Samples: {ml_stats.get('training_samples', 0)}")
        
        # Risk assessment
        if self.consecutive_losses > 0:
            print(f"   ⚠️ Consecutive Losses: {self.consecutive_losses}")
    
    async def _shutdown(self):
        """Shutdown swing trading bot with ML model saving"""
        
        print("\n🛑 Shutting down BTC swing trading bot...")
        self.is_running = False
        
        try:
            # Close any open positions
            position_info = self.trading_logic.get_position_info()
            if position_info['has_position']:
                print("🔄 Closing open swing position...")
                current_price = self.data_collector.get_current_price()
                self._close_swing_position({'close': current_price}, "Bot shutdown")
            
            # Save ML model
            if self.ml_interface:
                try:
                    self.ml_interface.ml_model.save_model()
                    print("🤖 Swing ML model saved for future learning")
                except Exception as e:
                    logging.warning(f"Error saving ML model: {e}")
            
            # Stop data feed
            self.data_collector.stop_data_feed()
            
            # Generate final report
            self._generate_final_report()
            
            # Cleanup logger
            self.trade_logger.cleanup()
            
            print("✅ BTC swing trading bot shutdown completed")
            
        except Exception as e:
            logging.error(f"Shutdown error: {e}")
    
    def _generate_final_report(self):
        """Generate final report with swing trading insights"""
        
        session_duration = datetime.now() - self.session_start
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        avg_hold = (self.total_hold_time / max(1, self.total_trades)) / 60  # minutes
        ml_accuracy = (self.ml_correct_today / max(1, self.ml_predictions_today)) * 100 if self.ml_predictions_today > 0 else 0
        
        print("\n" + "="*80)
        print("           ₿ BTC SWING TRADING SESSION FINAL REPORT - OPTIMIZED - DEBUG")
        print("="*80)
        
        # Session overview
        print(f"Session Duration: {str(session_duration).split('.')[0]}")
        print(f"Challenge Attempt: #{self.challenge_attempt}")
        print(f"Starting Balance: €{SWING_CONFIG['starting_balance']}")
        print(f"Final Balance: €{self.current_balance:.2f}")
        print(f"Confidence Threshold: {SWING_CONFIG['min_confidence']} (OPTIMIZED)")
        print(f"Debug Mode: ENABLED")
        
        # Swing Performance metrics
        balance_growth = ((self.current_balance - SWING_CONFIG['starting_balance']) / SWING_CONFIG['starting_balance']) * 100
        print(f"\n📊 SWING PERFORMANCE:")
        print(f"Balance Growth: {balance_growth:+.1f}%")
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Daily P&L: €{self.daily_pnl:+.2f}")
        print(f"Average Hold Time: {avg_hold:.1f} minutes")
        print(f"Position Multiplier: {SWING_CONFIG['position_multiplier']}x")
        print(f"Target Range: {SWING_CONFIG['min_position_time']//60}-{SWING_CONFIG['max_position_time']//60} minutes")
        
        # ML Learning insights
        if self.ml_enabled and self.ml_interface:
            ml_stats = self.ml_interface.get_ml_stats()
            print(f"\n🤖 SWING MACHINE LEARNING:")
            print(f"ML Predictions Today: {self.ml_predictions_today}")
            print(f"ML Accuracy Today: {ml_accuracy:.1f}%")
            print(f"Total Training Samples: {ml_stats.get('training_samples', 0)}")
            print(f"Model Version: v{ml_stats.get('model_version', 1)}")
            print(f"Overall ML Accuracy: {ml_stats.get('accuracy', 0):.1f}%")
            print(f"ML Confidence Threshold: {SWING_CONFIG['ml_min_confidence']} (OPTIMIZED)")
        
        # Challenge progress
        current_level = 0
        target = 20.0
        while target <= self.current_balance and target < 1000000:
            current_level += 1
            target *= 2
        
        print(f"\n🎯 SWING CHALLENGE PROGRESS:")
        print(f"Current Level: {current_level}")
        print(f"Distance to €1M: €{1000000 - self.current_balance:,.0f}")
        
        # Assessment
        if self.current_balance >= 1000000:
            print("🏆 SWING CHALLENGE COMPLETED! 🏆")
        elif win_rate >= 60 and avg_hold <= 5 and self.daily_pnl > 0:
            assessment = "🟢 EXCELLENT SWING SESSION!"
            if ml_accuracy > 70:
                assessment += " 🤖 ML LEARNING OPTIMALLY!"
        elif win_rate >= 50 and avg_hold <= 6:
            assessment = "🟡 GOOD SWING SESSION"
            if ml_accuracy > 60:
                assessment += " 🤖 ML IMPROVING"
        else:
            assessment = "🔴 LEARNING PHASE"
            if self.ml_enabled:
                assessment += " 🤖 ML GATHERING SWING DATA"
        
        print(f"\n{assessment}")
        
        # Optimization results
        print(f"\n💡 OPTIMIZATION RESULTS:")
        if self.total_trades > 0:
            print("   ✅ OPTIMIZATIONS SUCCESSFUL - Active trading achieved")
            print(f"   ✅ Signal generation working with {SWING_CONFIG['min_confidence']} threshold")
        else:
            print("   ⚠️ Consider further optimization if no trades occurred")
        
        # Debug results
        print(f"\n🐛 DEBUG ANALYSIS:")
        print("   ✅ Enhanced signal debugging enabled")
        print("   ✅ Candle processing monitoring active")
        print("   ✅ Trade execution path logging")
        
        print("="*80)


def nuclear_logic_test():
    """Nuclear logic test function - can be run independently"""
    
    print("\n🧪 NUCLEAR LOGIC TEST - Adding candles...")
    
    # Import the trading logic
    from trading_logic import BTCSwingLogic, SwingSignalType
    from datetime import datetime
    import time
    
    # Create test logic
    config = {
        'profit_target_pct': 2.5,
        'stop_loss_pct': 1.0,
        'min_confidence': 0.45,
        'signal_cooldown': 10
    }
    logic = BTCSwingLogic(config)
    
    # Add test candles to trigger nuclear logic
    for i in range(10):
        test_candle = {
            'timeframe': '1m',
            'timestamp': datetime.now(),
            'open': 43000 + i * 15,  # Rising price
            'high': 43000 + i * 15 + 25,
            'low': 43000 + i * 15 - 10,
            'close': 43000 + i * 15 + 12,  # Consistent upward movement
            'volume': 1.5 + i * 0.1,
            'body_size': 12,
            'is_bullish': True,
            'range': 35
        }
        
        print(f"🕯️ Adding candle {i+1}: €{test_candle['close']:.2f}")
        logic._update_candle_buffers(test_candle)
        
        # Test signal generation
        signal = logic._nuclear_signal_generation(test_candle, {})
        print(f"   Signal: {signal.signal_type.value} | Conf: {signal.confidence:.2f} | {signal.reasoning}")
        
        if signal.signal_type != SwingSignalType.HOLD:
            print(f"   🚀 SIGNAL GENERATED! {signal.signal_type.value.upper()}")
            break
            
        time.sleep(0.5)
    
    print("\n✅ Nuclear logic test completed!")


async def main():
    """Main entry point for BTC swing trading bot"""
    
    print("₿ BTC SWING TRADING BOT v3.0 - €20 to €1M Challenge - OPTIMIZED - DEBUG")
    print("=" * 70)
    print("SWING TRADING ENHANCEMENTS:")
    print("  ✅ HOLD TIMES: 2-5 minutes (vs 20 seconds)")
    print("  ✅ TARGETS: 2.5% percentage-based (vs fixed €8)")
    print("  ✅ STOPS: 1.0% percentage-based (vs fixed €4)")
    print("  ✅ ANALYSIS: Market structure + candles (vs raw ticks)")
    print("  ✅ POSITION SIZE: 1.5x sustainable (vs 3x aggressive)")
    print("  ✅ ML LEARNING: Swing pattern recognition")
    print()
    print("OPTIMIZATIONS APPLIED:")
    print("  🎯 CONFIDENCE: 0.45 (was 0.65) - More responsive")
    print("  🎯 ML THRESHOLD: 0.40 (was 0.60) - Enhanced ML")
    print("  🎯 SIGNAL COOLDOWN: 10s (was 30s) - Faster signals")
    print("  🎯 ML RETRAINING: 15 samples (was 20) - Faster learning")
    print("  🎯 DEBUG LOGGING: Enhanced signal analysis")
    print()
    print("DEBUG FEATURES:")
    print("  🐛 CANDLE PROCESSING: Real-time monitoring")
    print("  🐛 SIGNAL ANALYSIS: Always show confidence/reasoning")
    print("  🐛 EXECUTION PATH: Track signal → trade flow")
    print("  🐛 NUCLEAR TEST: Independent logic testing")
    print()
    print("ARCHITECTURE (5 Files):")
    print("  1. data_collection.py - BTC swing data with candles")
    print("  2. trading_logic.py - Swing strategy with market structure") 
    print("  3. trade_execution.py - Sustainable swing execution")
    print("  4. logger.py - Enhanced challenge tracking")
    print("  5. ml_interface.py - Swing pattern ML learning")
    print("  6. main.py - Complete swing integration (OPTIMIZED + DEBUG)")
    print()
    
    # Ask user if they want to run nuclear test first
    print("🧪 NUCLEAR LOGIC TEST AVAILABLE:")
    print("   Run this first if you want to test signal generation")
    print("   Type 'test' to run nuclear test, or press Enter to start bot")
    
    try:
        user_input = input(">>> ").strip().lower()
        if user_input == 'test':
            nuclear_logic_test()
            print("\nNuclear test completed. Starting main bot in 3 seconds...")
            await asyncio.sleep(3)
    except:
        pass  # Continue to main bot if input fails
    
    # Validate setup
    if not SWING_CONFIG['api_key'] or SWING_CONFIG['api_key'] == 'YOUR_ALPACA_API_KEY':
        print("⚠️ No API keys configured - will use simulation mode")
        print("💡 For live trading, add your Alpaca API keys to SWING_CONFIG")
        print()
    
    # ML status
    if SWING_CONFIG['ml_enabled']:
        print("🤖 Machine Learning: ENABLED (OPTIMIZED)")
        print("   • Learning from swing trade patterns")
        print("   • Market structure pattern recognition") 
        print("   • Multi-timeframe momentum analysis")
        print("   • Model auto-retraining every 15 samples (was 20)")
        print("   • ML confidence threshold: 0.40 (was 0.60)")
        print()
    else:
        print("🚫 Machine Learning: DISABLED")
        print()
    
    # Swing position sizing examples
    print("💰 SWING POSITION SIZING (1.5x Sustainable):")
    print("   € 20 account → ~0.00035 BTC = €15.00 position (sustainable)")
    print("   € 50 account → ~0.00053 BTC = €22.50 position (sustainable)")
    print("   €100 account → ~0.00075 BTC = €32.50 position (sustainable)")
    print("   €200 account → ~0.00107 BTC = €46.00 position (sustainable)")
    print()
    
    print("📊 SWING TRADING CAPABILITIES (OPTIMIZED + DEBUG):")
    print("   ✅ LONG positions (buy low, sell high)")
    print("   ✅ SHORT positions (sell high, buy low)")
    print("   ✅ Market structure analysis")
    print("   ✅ Multi-timeframe confirmation (1m + 3m)")
    print("   ✅ Support/resistance interaction")
    print("   ✅ Volume surge detection")
    print("   ✅ 1.5x sustainable position sizing")
    print("   ✅ Lower confidence thresholds for more activity")
    print("   ✅ Enhanced signal debugging")
    print("   ✅ Real-time candle processing monitoring")
    print()
    
    # Create and start swing bot
    bot = BTCSwingTradingBot()
    await bot.start_swing_trading()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 BTC Swing Trading Bot stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.error(f"Main error: {e}")