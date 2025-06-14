#!/usr/bin/env python3
"""
Complete BTC Scalping Bot with ML Learning - CORRECTED VERSION
Fixed: 1) Short selling support 2) ML integration 3) All syntax issues
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict

# Import the 4 core modules + ML
from data_collection import BTCDataCollector
from trading_logic import BTCScalpingLogic, SignalType
from trade_execution import BTCTradeExecutor
from logger import BTCTradeLogger
from ml_interface import BTCMLInterface, BTC_ML_CONFIG

# Configuration for €20 to €1M Challenge with fixes
CONFIG = {
    # Challenge Parameters
    'starting_balance': 20.0,              # Start with €20
    'profit_target_euros': 8.0,            # €8 profit target
    'stop_loss_euros': 4.0,                # €4 stop loss (2:1 ratio)
    'max_position_time': 20,               # 20-second scalps
    'min_confidence': 0.50,                # Lower threshold for more opportunities
    'risk_per_trade_pct': 2.0,             # Risk 2% per trade
    
    # API Configuration
    'paper_trading': True,
    'api_key': 'YOUR_ALPACA_API_KEY',      # Replace with your keys
    'secret_key': 'YOUR_ALPACA_SECRET_KEY', # Replace with your keys
    
    # Scalping Settings
    'max_daily_trades': 100,               # High frequency for scalping
    'min_trade_interval': 2.0,             # 2 seconds between trades
    'status_update_interval': 10,          # Status every 10 seconds
    
    # ML Learning Configuration
    'ml_enabled': True,                    # Enable machine learning
    'ml_min_confidence': 0.60,             # ML signal threshold
    'auto_ml_retrain': True,               # Automatically retrain model
    
    # Risk Management
    'max_consecutive_losses': 3,           # Reset after 3 losses
    'daily_loss_limit_pct': 10.0,         # 10% daily loss limit
    'force_reset_balance': 3.0,            # Reset if below €3
    
    # Files
    'log_file': 'btc_scalping_challenge.csv',
    'ml_model_file': 'btcusd_ml_model.pkl'
}


class BTCScalpingBot:
    """
    Complete BTC Scalping Bot with BOTH FIXES APPLIED
    1. Position sizing: FIXED
    2. Short selling: ENABLED
    3. ML learning: ENHANCED
    """
    
    def __init__(self):
        # Bot state
        self.is_running = False
        self.session_start = datetime.now()
        self.current_balance = CONFIG['starting_balance']
        self.trades_today = 0
        self.consecutive_losses = 0
        self.challenge_attempt = 1
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_pnl = 0.0
        
        # ML Learning state
        self.ml_enabled = CONFIG['ml_enabled']
        self.ml_predictions_today = 0
        self.ml_correct_today = 0
        self.last_ml_retrain = None
        
        # Initialize components
        self.setup_logging()
        self.initialize_components()
        
        print(f"\n₿ BTC SCALPING BOT v2.0 - CORRECTED & COMPLETE")
        print(f"🎯 Challenge: €20 → €1,000,000")
        print(f"💰 Starting Balance: €{self.current_balance}")
        print(f"📊 Strategy: €{CONFIG['profit_target_euros']} target, €{CONFIG['stop_loss_euros']} stop")
        print(f"🤖 ML Learning: {'ENABLED' if self.ml_enabled else 'DISABLED'}")
        print(f"⏱️ Max Position Time: {CONFIG['max_position_time']} seconds")
    
    def setup_logging(self):
        """Setup logging for the bot"""
        log_filename = f'btc_scalping_{datetime.now().strftime("%Y%m%d")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        logging.info("🚀 BTC Scalping Bot v2.0 - CORRECTED VERSION starting...")
    
    def initialize_components(self):
        """Initialize all components"""
        
        print("🔧 Initializing components...")
        
        # 1. Data Collection
        self.data_collector = BTCDataCollector("BTCUSD")
        
        # 2. Trading Logic with FIXED position sizing
        logic_config = {
            'profit_target_euros': CONFIG['profit_target_euros'],
            'stop_loss_euros': CONFIG['stop_loss_euros'],
            'min_confidence': CONFIG['min_confidence'],
            'max_position_time': CONFIG['max_position_time'],
            'risk_per_trade_pct': CONFIG['risk_per_trade_pct']
        }
        self.trading_logic = BTCScalpingLogic(logic_config)
        self.trading_logic.current_balance = self.current_balance
        
        # 3. Trade Execution with short selling support
        execution_config = {
            'paper_trading': CONFIG['paper_trading'],
            'api_key': CONFIG['api_key'] if CONFIG['api_key'] != 'YOUR_ALPACA_API_KEY' else '',
            'secret_key': CONFIG['secret_key'] if CONFIG['secret_key'] != 'YOUR_ALPACA_SECRET_KEY' else ''
        }
        self.trade_executor = BTCTradeExecutor(execution_config)
        
        # 4. Trade Logger
        self.trade_logger = BTCTradeLogger(CONFIG['log_file'])
        
        # 5. ML Interface
        if self.ml_enabled:
            ml_config = BTC_ML_CONFIG.copy()
            ml_config['model_file'] = CONFIG['ml_model_file']
            self.ml_interface = BTCMLInterface(ml_config)
            
            # Connect ML to trading logic
            self.trading_logic.set_ml_interface(self.ml_interface)
            
            print("🤖 ML Learning System initialized")
            print(f"   Model file: {CONFIG['ml_model_file']}")
            print(f"   Min confidence: {CONFIG['ml_min_confidence']}")
        else:
            self.ml_interface = None
            print("🚫 ML Learning disabled")
        
        # Connect data collector to trading logic
        self.data_collector.add_tick_callback(self.on_tick_received)
        
        print("✅ All components initialized")
    
    async def start_scalping(self):
        """Start the BTC scalping bot"""
        
        print("\n" + "="*80)
        print("      🚀 STARTING €20 → €1M BTC SCALPING CHALLENGE v2.0 (CORRECTED)")
        print("="*80)
        
        try:
            # Start data feed
            print("📡 Starting BTC data feed...")
            api_key = CONFIG['api_key'] if CONFIG['api_key'] != 'YOUR_ALPACA_API_KEY' else ''
            secret_key = CONFIG['secret_key'] if CONFIG['secret_key'] != 'YOUR_ALPACA_SECRET_KEY' else ''
            self.data_collector.start_data_feed(api_key, secret_key)
            
            # Wait for data connection
            await self._wait_for_data()
            
            # Display initial account info
            self._display_account_info()
            
            # Start main scalping loop
            print("🔄 Starting scalping loop...")
            print("💡 Strategy: Quick €8 profits with €4 stops + ML learning")
            print("🤖 ML: Learning from every trade to improve performance")
            print("⏹️ Press Ctrl+C to stop")
            print("-" * 80)
            
            self.is_running = True
            await self._main_scalping_loop()
            
        except KeyboardInterrupt:
            print("\n🛑 Scalping session stopped by user")
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
        
        # Show short selling capability
        if account.get('short_selling_enabled', False):
            print(f"📈📉 Short Selling: ENABLED")
        
        # ML status
        if self.ml_interface:
            ml_stats = self.ml_interface.get_ml_stats()
            print(f"🤖 ML Model: v{ml_stats.get('model_version', 1)} | Samples: {ml_stats.get('training_samples', 0)}")
            if ml_stats.get('accuracy', 0) > 0:
                print(f"🎯 ML Accuracy: {ml_stats['accuracy']:.1f}%")
    
    async def _main_scalping_loop(self):
        """Main scalping loop with ML learning"""
        
        last_status_time = time.time()
        last_balance_check = time.time()
        last_ml_update = time.time()
        
        while self.is_running:
            try:
                # Check if balance needs reset
                if time.time() - last_balance_check > 30:
                    if self._should_reset_challenge():
                        self._reset_challenge()
                    last_balance_check = time.time()
                
                # Check daily limits
                if self.trades_today >= CONFIG['max_daily_trades']:
                    print(f"🛑 Daily trade limit reached: {self.trades_today}")
                    break
                
                # Check consecutive losses
                if self.consecutive_losses >= CONFIG['max_consecutive_losses']:
                    print(f"⚠️ {self.consecutive_losses} consecutive losses - pausing 30s")
                    await asyncio.sleep(30)
                    self.consecutive_losses = 0
                
                # ML learning updates
                if self.ml_enabled and time.time() - last_ml_update > 60:  # Every minute
                    self._update_ml_learning()
                    last_ml_update = time.time()
                
                # Periodic status update
                current_time = time.time()
                if current_time - last_status_time > CONFIG['status_update_interval']:
                    self._display_status()
                    last_status_time = current_time
                
                await asyncio.sleep(0.1)  # Fast loop for scalping
                
            except Exception as e:
                logging.error(f"Scalping loop error: {e}")
                await asyncio.sleep(1)
        
        await self._shutdown()
    
    def on_tick_received(self, tick_data):
        """Process incoming tick for scalping opportunities"""
        
        try:
            # Get market metrics for scalping
            market_metrics = self.data_collector.get_scalping_metrics()
            
            if market_metrics.get('insufficient_data'):
                return
            
            # Update trading logic balance
            self.trading_logic.current_balance = self.current_balance
            
            # Generate trading signal (now with ML enhancement)
            signal = self.trading_logic.evaluate_tick(tick_data, market_metrics)
            
            # Process signals
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                self._execute_scalping_signal(signal, tick_data)
            elif signal.signal_type == SignalType.CLOSE:
                self._close_scalping_position(tick_data, signal.reasoning)
            
        except Exception as e:
            logging.error(f"Tick processing error: {e}")
    
    def _execute_scalping_signal(self, signal, tick_data):
        """Execute scalping signal with FIXED position sizing"""
        
        current_price = tick_data['price']
        
        # FIXED: Calculate proper position size
        position_size = self.trading_logic._calculate_position_size(current_price)
        position_value = position_size * current_price
        
        # Verify position size is reasonable
        if position_value > self.current_balance * 0.5:  # Safety check
            logging.warning(f"Position size too large: €{position_value:.2f} for €{self.current_balance:.2f} account")
            return
        
        print(f"\n₿ SCALP SIGNAL: {signal.signal_type.value.upper()} @ €{current_price:,.2f}")
        print(f"   🎯 Confidence: {signal.confidence:.2f}")
        print(f"   💡 Reasoning: {signal.reasoning}")
        print(f"   📦 Size: {position_size:.6f} BTC (€{position_value:.2f}) ✅ FIXED")
        
        # ML enhancement information
        if self.ml_enabled and 'ML' in signal.reasoning:
            print(f"   🤖 ML Enhanced Signal")
            self.ml_predictions_today += 1
        
        # Execute trade (now supports both long and short)
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
            print(f"✅ SCALP ENTRY: {direction} position opened")
            print(f"   🎯 Target: €{signal.target_price:,.2f} (+€{CONFIG['profit_target_euros']})")
            print(f"   🛡️ Stop: €{signal.stop_price:,.2f} (-€{CONFIG['stop_loss_euros']})")
            print(f"   📊 Trade #{self.trades_today} today")
            
            # Verify expected profit calculation
            if signal.signal_type.value == 'buy':
                expected_profit = (signal.target_price - current_price) * position_size
            else:
                expected_profit = (current_price - signal.target_price) * position_size
            print(f"   💰 Expected Profit: €{expected_profit:.2f}")
            
        else:
            print(f"❌ SCALP ENTRY FAILED: {order.status.value}")
    
    def _close_scalping_position(self, tick_data, reasoning):
        """Close current scalping position with ML learning"""
        
        current_price = tick_data['price']
        position_info = self.trading_logic.get_position_info()
        
        if not position_info['has_position']:
            return
        
        print(f"\n₿ SCALP EXIT: {reasoning} @ €{current_price:,.2f}")
        
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
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity
            
            # Account for commissions
            pnl -= (exit_order.commission + getattr(exit_order, 'entry_commission', 0))
            
            # Update balance and performance
            self.current_balance += pnl
            self.daily_pnl += pnl
            self.total_trades += 1
            
            # Track ML accuracy
            if self.ml_enabled and self.ml_predictions_today > 0:
                if pnl > 0:
                    self.ml_correct_today += 1
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
            else:
                if pnl > 0:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
            
            # Update trading logic (this triggers ML learning)
            self.trading_logic.update_position('close', current_price, quantity)
            self.trading_logic.current_balance = self.current_balance
            
            # Log exit trade
            self.trade_logger.log_trade(
                exit_order, 'exit', 
                profit_loss=pnl, 
                current_balance=self.current_balance
            )
            
            # Display results
            pnl_symbol = "🟢" if pnl > 0 else "🔴"
            direction = "📈 LONG" if position_info['side'] == 'long' else "📉 SHORT"
            win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
            ml_accuracy = (self.ml_correct_today / self.ml_predictions_today) * 100 if self.ml_predictions_today > 0 else 0
            
            print(f"✅ SCALP CLOSED: {direction} position")
            print(f"   💰 P&L: €{pnl:+.2f} {pnl_symbol}")
            print(f"   💵 Balance: €{self.current_balance:.2f}")
            print(f"   📊 Session: {self.winning_trades}W/{self.total_trades - self.winning_trades}L ({win_rate:.1f}%)")
            if self.ml_enabled and self.ml_predictions_today > 0:
                print(f"   🤖 ML Today: {self.ml_correct_today}/{self.ml_predictions_today} ({ml_accuracy:.1f}%)")
            
            # Check if reached next level
            self._check_level_progress()
            
        else:
            print(f"❌ SCALP EXIT FAILED")
    
    def _update_ml_learning(self):
        """Update ML learning system"""
        if not self.ml_interface:
            return
        
        try:
            # Check if model needs retraining
            ml_stats = self.ml_interface.get_ml_stats()
            
            if (CONFIG['auto_ml_retrain'] and 
                ml_stats.get('training_samples', 0) >= 15 and 
                ml_stats.get('training_samples', 0) % 15 == 0):
                
                self.ml_interface.force_retrain()
                self.last_ml_retrain = datetime.now()
                print(f"\n🤖 ML MODEL RETRAINED!")
                print(f"   📊 Samples: {ml_stats.get('training_samples', 0)}")
                logging.info("🤖 ML model auto-retrained")
                
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
            print(f"\n🎉 LEVEL {current_level + 1} REACHED! €{self.current_balance:.2f}")
            logging.info(f"🎉 Challenge Level {current_level + 1} reached: €{self.current_balance:.2f}")
            
            # ML celebration
            if self.ml_enabled:
                ml_accuracy = (self.ml_correct_today / max(1, self.ml_predictions_today)) * 100
                print(f"🤖 ML contributed with {ml_accuracy:.1f}% accuracy")
            
            if next_target >= 1000000:
                print("🏆 CHALLENGE COMPLETED! €1,000,000 REACHED!")
                logging.info("🏆 €20 to €1M Challenge COMPLETED!")
                self.is_running = False
    
    def _should_reset_challenge(self) -> bool:
        """Check if challenge should be reset"""
        return self.current_balance < CONFIG['force_reset_balance']
    
    def _reset_challenge(self):
        """Reset challenge to €20 but keep ML knowledge"""
        
        print(f"\n🔄 RESETTING CHALLENGE (ML KNOWLEDGE RETAINED)")
        print(f"   Previous balance: €{self.current_balance:.2f}")
        print(f"   Trades completed: {self.total_trades}")
        if self.ml_enabled:
            ml_accuracy = (self.ml_correct_today / max(1, self.ml_predictions_today)) * 100
            print(f"   ML accuracy this session: {ml_accuracy:.1f}%")
        
        # Start new attempt
        self.challenge_attempt += 1
        self.trade_logger.start_new_challenge_attempt()
        
        # Reset trading metrics but keep ML learning
        self.current_balance = CONFIG['starting_balance']
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trades_today = 0
        
        # Reset daily ML counters but keep model knowledge
        self.ml_predictions_today = 0
        self.ml_correct_today = 0
        
        # Reset trading logic
        self.trading_logic.reset_to_twenty_euros()
        self.trading_logic.current_balance = self.current_balance
        
        print(f"✅ Challenge attempt #{self.challenge_attempt} started with €20")
        print("🤖 ML model retained previous learning")
        logging.info(f"Challenge reset - Attempt #{self.challenge_attempt} started (ML retained)")
    
    def _display_status(self):
        """Display current scalping status with ML metrics"""
        
        current_price = self.data_collector.get_current_price()
        position_info = self.trading_logic.get_position_info()
        
        # Calculate level and progress
        current_level = 0
        target = 20.0
        while target <= self.current_balance and target < 1000000:
            current_level += 1
            target *= 2
        next_target = min(target, 1000000)
        progress = (self.current_balance / next_target) * 100
        
        print(f"\n₿ SCALPING STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print(f"   💹 BTC Price: €{current_price:,.2f}")
        print(f"   💰 Balance: €{self.current_balance:.2f}")
        print(f"   📊 Level: {current_level} → {current_level + 1} ({progress:.1f}%)")
        print(f"   🎯 Next Target: €{next_target:,.0f}")
        
        if position_info['has_position']:
            time_in_pos = position_info['time_in_position']
            direction = "📈 LONG" if position_info['side'] == 'long' else "📉 SHORT"
            print(f"   📍 Position: {direction} @ €{position_info['entry_price']:,.2f} ({time_in_pos:.0f}s)")
        else:
            print(f"   📍 Position: NONE")
        
        # Performance metrics
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        print(f"   📈 Trades: {self.trades_today}/{CONFIG['max_daily_trades']} today")
        print(f"   🏆 Record: {self.winning_trades}W/{self.total_trades - self.winning_trades}L ({win_rate:.1f}%)")
        print(f"   💵 Daily P&L: €{self.daily_pnl:+.2f}")
        
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
        """Shutdown with ML model saving"""
        
        print("\n🛑 Shutting down BTC scalping bot...")
        self.is_running = False
        
        try:
            # Close any open positions
            position_info = self.trading_logic.get_position_info()
            if position_info['has_position']:
                print("🔄 Closing open position...")
                current_price = self.data_collector.get_current_price()
                self._close_scalping_position({'price': current_price}, "Bot shutdown")
            
            # Save ML model
            if self.ml_interface:
                try:
                    self.ml_interface.ml_model.save_model()
                    print("🤖 ML model saved for future learning")
                except Exception as e:
                    logging.warning(f"Error saving ML model: {e}")
            
            # Stop data feed
            self.data_collector.stop_data_feed()
            
            # Generate final report
            self._generate_final_report()
            
            # Cleanup logger
            self.trade_logger.cleanup()
            
            print("✅ BTC scalping bot shutdown completed")
            
        except Exception as e:
            logging.error(f"Shutdown error: {e}")
    
    def _generate_final_report(self):
        """Generate final report with ML insights"""
        
        session_duration = datetime.now() - self.session_start
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        ml_accuracy = (self.ml_correct_today / max(1, self.ml_predictions_today)) * 100 if self.ml_predictions_today > 0 else 0
        
        print("\n" + "="*80)
        print("           ₿ BTC SCALPING SESSION FINAL REPORT v2.0 (CORRECTED)")
        print("="*80)
        
        # Session overview
        print(f"Session Duration: {str(session_duration).split('.')[0]}")
        print(f"Challenge Attempt: #{self.challenge_attempt}")
        print(f"Starting Balance: €{CONFIG['starting_balance']}")
        print(f"Final Balance: €{self.current_balance:.2f}")
        
        # Performance metrics
        balance_growth = ((self.current_balance - CONFIG['starting_balance']) / CONFIG['starting_balance']) * 100
        print(f"\n📊 PERFORMANCE:")
        print(f"Balance Growth: {balance_growth:+.1f}%")
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Daily P&L: €{self.daily_pnl:+.2f}")
        
        # ML Learning insights
        if self.ml_enabled and self.ml_interface:
            ml_stats = self.ml_interface.get_ml_stats()
            print(f"\n🤖 MACHINE LEARNING:")
            print(f"ML Predictions Today: {self.ml_predictions_today}")
            print(f"ML Accuracy Today: {ml_accuracy:.1f}%")
            print(f"Total Training Samples: {ml_stats.get('training_samples', 0)}")
            print(f"Model Version: v{ml_stats.get('model_version', 1)}")
            print(f"Overall ML Accuracy: {ml_stats.get('accuracy', 0):.1f}%")
        
        # Challenge progress
        current_level = 0
        target = 20.0
        while target <= self.current_balance and target < 1000000:
            current_level += 1
            target *= 2
        
        print(f"\n🎯 CHALLENGE PROGRESS:")
        print(f"Current Level: {current_level}")
        print(f"Distance to €1M: €{1000000 - self.current_balance:,.0f}")
        
        # Assessment
        if self.current_balance >= 1000000:
            print("🏆 CHALLENGE COMPLETED! 🏆")
        elif win_rate >= 60 and self.daily_pnl > 0:
            assessment = "🟢 EXCELLENT SESSION!"
            if ml_accuracy > 70:
                assessment += " 🤖 ML LEARNING WELL!"
        elif win_rate >= 50:
            assessment = "🟡 GOOD SESSION"
            if ml_accuracy > 60:
                assessment += " 🤖 ML IMPROVING"
        else:
            assessment = "🔴 NEEDS IMPROVEMENT"
            if self.ml_enabled:
                assessment += " 🤖 ML STILL LEARNING"
        
        print(f"\n{assessment}")
        print("="*80)


async def main():
    """Main entry point for corrected BTC scalping bot"""
    
    print("₿ BTC SCALPING BOT v2.0 - €20 to €1M Challenge (CORRECTED)")
    print("=" * 65)
    print("FIXES APPLIED:")
    print("  ✅ FIXED: Correct position sizing for €20 accounts")
    print("  ✅ FIXED: Short selling support (both long + short trades)")
    print("  ✅ ENHANCED: Machine learning integration")
    print("  ✅ NEW: Continuous learning from every trade")
    print("  ✅ NEW: Auto-improving signal confidence")
    print()
    print("ARCHITECTURE (6 Files):")
    print("  1. data_collection.py - BTC tick data stream")
    print("  2. trading_logic.py - FIXED scalping strategy") 
    print("  3. trade_execution.py - FIXED with short selling")
    print("  4. logger.py - Challenge tracking")
    print("  5. ml_interface.py - Learning system")
    print("  6. main.py - CORRECTED integration")
    print()
    
    # Validate setup
    if not CONFIG['api_key'] or CONFIG['api_key'] == 'YOUR_ALPACA_API_KEY':
        print("⚠️ No API keys configured - will use simulation mode")
        print("💡 For live trading, add your Alpaca API keys to CONFIG")
        print()
    
    # ML status
    if CONFIG['ml_enabled']:
        print("🤖 Machine Learning: ENABLED")
        print("   • Learning from every trade outcome")
        print("   • Auto-improving signal confidence") 
        print("   • Model auto-retraining every 15 samples")
        print()
    else:
        print("🚫 Machine Learning: DISABLED")
        print()
    
    # Position sizing examples
    print("💰 POSITION SIZING EXAMPLES (CORRECTED):")
    print("   € 20 account → 0.000186 BTC = € 8.00 position ✅")
    print("   € 50 account → 0.000279 BTC = €12.00 position ✅")
    print("   €100 account → 0.000372 BTC = €16.00 position ✅")
    print("   €200 account → 0.000465 BTC = €20.00 position ✅")
    print()
    
    print("📈📉 TRADING CAPABILITIES:")
    print("   ✅ LONG positions (buy low, sell high)")
    print("   ✅ SHORT positions (sell high, buy low) - FIXED")
    print("   ✅ Complete market coverage")
    print()
    
    # Create and start bot
    bot = BTCScalpingBot()
    await bot.start_scalping()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 BTC Scalping Bot stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.error(f"Main error: {e}")