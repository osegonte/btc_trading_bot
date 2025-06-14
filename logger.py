#!/usr/bin/env python3
"""
BTC Trade Logger - Core File 4/4
Purpose: Log trades and track €20 to €1M challenge progress
Essential for analysis and strategy improvement
"""

import csv
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class BTCTradeRecord:
    """Individual BTC trade record for the challenge"""
    def __init__(self, trade, trade_type: str = "entry", profit_loss: float = 0.0):
        self.timestamp = trade.timestamp
        self.symbol = trade.symbol
        self.side = trade.side
        self.quantity = trade.quantity
        self.price = trade.fill_price or trade.price
        self.order_id = trade.order_id
        self.status = trade.status.value
        self.trade_type = trade_type  # 'entry' or 'exit'
        self.profit_loss = profit_loss
        self.commission = getattr(trade, 'commission', 0.0)
        self.slippage = getattr(trade, 'slippage', 0.0)


class ChallengeTracker:
    """Track progress in €20 to €1M challenge"""
    def __init__(self):
        self.challenge_attempts = []
        self.current_attempt = {
            'start_time': datetime.now(),
            'start_balance': 20.0,
            'current_balance': 20.0,
            'trades': [],
            'peak_balance': 20.0,
            'max_drawdown': 0.0,
            'attempt_number': 1
        }
    
    def start_new_attempt(self, attempt_number: int):
        """Start a new challenge attempt"""
        if self.current_attempt['trades']:
            self.challenge_attempts.append(self.current_attempt.copy())
        
        self.current_attempt = {
            'start_time': datetime.now(),
            'start_balance': 20.0,
            'current_balance': 20.0,
            'trades': [],
            'peak_balance': 20.0,
            'max_drawdown': 0.0,
            'attempt_number': attempt_number
        }
    
    def update_balance(self, new_balance: float, trade_pnl: float):
        """Update current balance and track metrics"""
        self.current_attempt['current_balance'] = new_balance
        
        # Track peak balance
        if new_balance > self.current_attempt['peak_balance']:
            self.current_attempt['peak_balance'] = new_balance
        
        # Track max drawdown
        current_drawdown = self.current_attempt['peak_balance'] - new_balance
        if current_drawdown > self.current_attempt['max_drawdown']:
            self.current_attempt['max_drawdown'] = current_drawdown
    
    def get_current_level(self) -> int:
        """Calculate current level in €20 to €1M challenge"""
        balance = self.current_attempt['current_balance']
        level = 0
        target = 20.0
        
        while target <= balance and target < 1000000:
            level += 1
            target *= 2  # Double each level
        
        return level
    
    def get_next_target(self) -> float:
        """Get next balance target"""
        level = self.get_current_level()
        return min(20.0 * (2 ** level), 1000000)
    
    def get_progress_summary(self) -> Dict:
        """Get comprehensive progress summary"""
        current_level = self.get_current_level()
        next_target = self.get_next_target()
        
        # Calculate growth rate
        balance = self.current_attempt['current_balance']
        start_balance = self.current_attempt['start_balance']
        growth_rate = ((balance - start_balance) / start_balance) * 100
        
        # Calculate time in current attempt
        time_elapsed = datetime.now() - self.current_attempt['start_time']
        
        return {
            'attempt_number': self.current_attempt['attempt_number'],
            'current_balance': balance,
            'current_level': current_level,
            'next_target': next_target,
            'progress_to_next': (balance / next_target) * 100,
            'growth_rate': growth_rate,
            'time_elapsed': str(time_elapsed).split('.')[0],
            'peak_balance': self.current_attempt['peak_balance'],
            'max_drawdown': self.current_attempt['max_drawdown'],
            'distance_to_million': 1000000 - balance,
            'total_attempts': len(self.challenge_attempts) + 1
        }


class BTCTradeLogger:
    """
    Comprehensive BTC trade logger for €20 to €1M challenge
    Tracks all trades, performance metrics, and challenge progress
    """
    
    def __init__(self, log_file: str = "btc_scalping_challenge.csv"):
        self.log_file = log_file
        self.trades = []
        
        # Challenge tracking
        self.challenge_tracker = ChallengeTracker()
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.session_start = datetime.now()
        
        # BTC-specific metrics
        self.largest_win = 0.0
        self.largest_loss = 0.0
        self.total_btc_traded = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        # Setup files
        self._setup_csv_file()
        self._setup_challenge_log()
        
        logging.info(f"✅ BTC Trade Logger initialized - Challenge tracking active")
    
    def _setup_csv_file(self):
        """Setup CSV file with headers"""
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'price',
                    'order_id', 'status', 'trade_type', 'profit_loss', 'commission',
                    'slippage', 'balance_after', 'challenge_level', 'attempt_number'
                ])
            logging.info(f"Created new challenge CSV: {self.log_file}")
        else:
            self._load_existing_trades()
    
    def _setup_challenge_log(self):
        """Setup challenge progress log"""
        self.challenge_log_file = self.log_file.replace('.csv', '_challenge.json')
        
        if os.path.exists(self.challenge_log_file):
            try:
                with open(self.challenge_log_file, 'r') as f:
                    challenge_data = json.load(f)
                    self.challenge_tracker.challenge_attempts = challenge_data.get('attempts', [])
                    
                    # Load current attempt if exists
                    current = challenge_data.get('current_attempt')
                    if current:
                        current['start_time'] = datetime.fromisoformat(current['start_time'])
                        self.challenge_tracker.current_attempt = current
            except Exception as e:
                logging.warning(f"Could not load challenge data: {e}")
    
    def _load_existing_trades(self):
        """Load existing trades from CSV"""
        try:
            with open(self.log_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Update performance metrics
                    if row.get('trade_type') == 'exit' and row.get('profit_loss'):
                        pnl = float(row['profit_loss'])
                        self.total_trades += 1
                        self.total_pnl += pnl
                        
                        if pnl > 0:
                            self.winning_trades += 1
                            self.largest_win = max(self.largest_win, pnl)
                        else:
                            self.largest_loss = min(self.largest_loss, pnl)
                    
                    # Track volume and costs
                    if row.get('quantity'):
                        self.total_btc_traded += float(row['quantity'])
                    if row.get('commission'):
                        self.total_commission += float(row['commission'])
                    if row.get('slippage'):
                        self.total_slippage += float(row['slippage'])
            
            logging.info(f"Loaded {self.total_trades} existing trades")
            
        except Exception as e:
            logging.error(f"Error loading trades: {e}")
    
    def log_trade(self, trade, trade_type: str = "entry", profit_loss: float = 0.0, 
                  current_balance: float = 0.0):
        """Log a BTC trade with challenge tracking"""
        
        # Create trade record
        record = BTCTradeRecord(trade, trade_type, profit_loss)
        self.trades.append(record)
        
        # Update challenge tracker
        if trade_type == "exit" and profit_loss != 0:
            self.challenge_tracker.update_balance(current_balance, profit_loss)
            
            # Update performance metrics
            self.total_trades += 1
            self.total_pnl += profit_loss
            
            if profit_loss > 0:
                self.winning_trades += 1
                self.largest_win = max(self.largest_win, profit_loss)
            else:
                self.largest_loss = min(self.largest_loss, profit_loss)
        
        # Track volume and costs
        self.total_btc_traded += record.quantity
        self.total_commission += record.commission
        self.total_slippage += record.slippage
        
        # Write to CSV
        self._write_to_csv(record, current_balance)
        
        # Save challenge progress
        self._save_challenge_progress()
        
        # Log the trade
        pnl_str = f" | P&L: €{profit_loss:+.2f}" if profit_loss != 0 else ""
        btc_value = record.quantity * record.price
        
        logging.info(f"₿ TRADE [{trade_type.upper()}]: {record.side.upper()} {record.quantity:.6f} BTC @ €{record.price:,.2f} (€{btc_value:,.2f}){pnl_str}")
        
        # Challenge progress update
        if trade_type == "exit":
            progress = self.challenge_tracker.get_progress_summary()
            logging.info(f"💰 Balance: €{current_balance:.2f} | Level: {progress['current_level']} | Target: €{progress['next_target']:.0f}")
    
    def _write_to_csv(self, record: BTCTradeRecord, current_balance: float):
        """Write trade record to CSV file"""
        
        try:
            progress = self.challenge_tracker.get_progress_summary()
            
            with open(self.log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    record.timestamp,
                    record.symbol,
                    record.side,
                    record.quantity,
                    record.price,
                    record.order_id,
                    record.status,
                    record.trade_type,
                    record.profit_loss,
                    record.commission,
                    record.slippage,
                    current_balance,
                    progress['current_level'],
                    progress['attempt_number']
                ])
        except Exception as e:
            logging.error(f"Error writing to CSV: {e}")
    
    def _save_challenge_progress(self):
        """Save challenge progress to JSON file"""
        
        try:
            # Prepare data for JSON serialization
            current_attempt = self.challenge_tracker.current_attempt.copy()
            current_attempt['start_time'] = current_attempt['start_time'].isoformat()
            
            challenge_data = {
                'current_attempt': current_attempt,
                'attempts': self.challenge_tracker.challenge_attempts,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.challenge_log_file, 'w') as f:
                json.dump(challenge_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving challenge progress: {e}")
    
    def start_new_challenge_attempt(self):
        """Start a new €20 to €1M challenge attempt"""
        
        current_attempt = self.challenge_tracker.current_attempt['attempt_number']
        new_attempt = current_attempt + 1
        
        logging.info(f"🔄 Starting new challenge attempt #{new_attempt}")
        
        # Log summary of previous attempt
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            logging.info(f"   Previous attempt: {self.total_trades} trades, {win_rate:.1f}% win rate, €{self.total_pnl:+.2f} P&L")
        
        # Reset tracking
        self.challenge_tracker.start_new_attempt(new_attempt)
        
        # Reset performance metrics for new attempt
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.largest_win = 0.0
        self.largest_loss = 0.0
        
        self._save_challenge_progress()
        
        logging.info(f"✅ Challenge attempt #{new_attempt} started with €20")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        avg_pnl = self.total_pnl / max(1, self.total_trades)
        session_duration = datetime.now() - self.session_start
        
        # Calculate profit factor
        wins = [t.profit_loss for t in self.trades if t.trade_type == 'exit' and t.profit_loss > 0]
        losses = [t.profit_loss for t in self.trades if t.trade_type == 'exit' and t.profit_loss < 0]
        
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.total_trades - self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'average_pnl': avg_pnl,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'profit_factor': profit_factor,
            'total_btc_traded': self.total_btc_traded,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'session_duration': str(session_duration).split('.')[0],
            'trades_logged': len(self.trades)
        }
    
    def get_challenge_summary(self) -> Dict:
        """Get €20 to €1M challenge summary"""
        
        progress = self.challenge_tracker.get_progress_summary()
        performance = self.get_performance_summary()
        
        # Calculate additional challenge metrics
        balance = progress['current_balance']
        time_to_million = "∞"
        
        if performance['average_pnl'] > 0 and self.total_trades > 0:
            # Estimate time to million based on current performance
            remaining = 1000000 - balance
            trades_needed = remaining / performance['average_pnl']
            # Assume 1 trade per minute on average
            hours_needed = trades_needed / 60
            time_to_million = f"{hours_needed:.1f} hours"
        
        return {
            **progress,
            **performance,
            'estimated_time_to_million': time_to_million,
            'success_probability': min(100, performance['win_rate'] * (performance['profit_factor'] / 2)),
            'risk_level': 'Low' if performance['profit_factor'] > 2 else 'Medium' if performance['profit_factor'] > 1.5 else 'High'
        }
    
    def print_challenge_status(self):
        """Print current challenge status"""
        
        summary = self.get_challenge_summary()
        
        print(f"\n₿ €20 → €1M CHALLENGE STATUS")
        print("=" * 50)
        print(f"Attempt: #{summary['attempt_number']}")
        print(f"Current Balance: €{summary['current_balance']:,.2f}")
        print(f"Level: {summary['current_level']}")
        print(f"Next Target: €{summary['next_target']:,.0f}")
        print(f"Progress: {summary['progress_to_next']:.1f}%")
        print(f"Distance to €1M: €{summary['distance_to_million']:,.0f}")
        print(f"Growth Rate: {summary['growth_rate']:+.1f}%")
        print(f"Time Elapsed: {summary['time_elapsed']}")
        
        print(f"\n📊 PERFORMANCE")
        print(f"Trades: {summary['total_trades']}")
        print(f"Win Rate: {summary['win_rate']:.1f}%")
        print(f"Total P&L: €{summary['total_pnl']:+.2f}")
        print(f"Profit Factor: {summary['profit_factor']:.2f}")
        print(f"Risk Level: {summary['risk_level']}")
        
        if summary['estimated_time_to_million'] != "∞":
            print(f"Est. Time to €1M: {summary['estimated_time_to_million']}")
        
        print("=" * 50)
    
    def export_challenge_data(self, filename: Optional[str] = None) -> str:
        """Export all challenge data to JSON file"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"btc_challenge_export_{timestamp}.json"
        
        try:
            export_data = {
                'challenge_summary': self.get_challenge_summary(),
                'performance_summary': self.get_performance_summary(),
                'all_attempts': self.challenge_tracker.challenge_attempts,
                'current_attempt': self.challenge_tracker.current_attempt,
                'recent_trades': [
                    {
                        'timestamp': t.timestamp,
                        'side': t.side,
                        'quantity': t.quantity,
                        'price': t.price,
                        'type': t.trade_type,
                        'pnl': t.profit_loss
                    }
                    for t in self.trades[-20:]  # Last 20 trades
                ],
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logging.info(f"✅ Challenge data exported to: {filename}")
            return filename
            
        except Exception as e:
            logging.error(f"Error exporting challenge data: {e}")
            return ""
    
    def cleanup(self):
        """Clean up and save final challenge state"""
        
        try:
            # Save final challenge progress
            self._save_challenge_progress()
            
            # Print final summaries
            self.print_challenge_status()
            
            # Export final data
            export_file = self.export_challenge_data()
            if export_file:
                print(f"\n💾 Challenge data exported to: {export_file}")
            
            logging.info("✅ BTC Trade Logger cleanup completed")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    # Test BTC trade logger
    logger = BTCTradeLogger()
    
    # Mock trade object
    class MockTrade:
        def __init__(self, side: str, price: float):
            self.timestamp = datetime.now().isoformat()
            self.symbol = 'BTCUSD'
            self.side = side
            self.quantity = 0.001
            self.price = price
            self.fill_price = price + (1 if side == 'buy' else -1)
            self.order_id = f'test_{side}_{int(datetime.now().timestamp())}'
            self.status = type('Status', (), {'value': 'filled'})()
            self.commission = 0.50
            self.slippage = 1.00
    
    print("🧪 Testing BTC Trade Logger...")
    
    # Simulate some trades
    entry_trade = MockTrade('buy', 43250.00)
    logger.log_trade(entry_trade, 'entry', current_balance=20.0)
    
    exit_trade = MockTrade('sell', 43258.00)
    logger.log_trade(exit_trade, 'exit', profit_loss=8.0, current_balance=28.0)
    
    # Print status
    logger.print_challenge_status()
    
    # Test export
    export_file = logger.export_challenge_data()
    print(f"Exported to: {export_file}")
    
    # Cleanup
    logger.cleanup()
    
    print("✅ BTC Trade Logger test completed")