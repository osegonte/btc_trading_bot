#!/usr/bin/env python3
"""
BTC Swing Challenge Logger - €20 to €1M Progress Tracking
Purpose: Track swing trades and challenge progression with enhanced metrics
Key Changes: Tick logging → Swing trade analysis with hold time tracking
"""

import csv
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SwingTradeRecord:
    """Individual swing trade record for challenge tracking"""
    def __init__(self, trade, trade_type: str = "entry", profit_loss: float = 0.0, hold_time: int = 0):
        self.timestamp = trade.timestamp
        self.symbol = trade.symbol
        self.side = trade.side
        self.quantity = trade.quantity
        self.price = trade.fill_price or trade.price
        self.order_id = trade.order_id
        self.status = trade.status.value
        self.trade_type = trade_type  # 'entry' or 'exit'
        self.profit_loss = profit_loss
        self.hold_time = hold_time  # seconds
        self.commission = getattr(trade, 'commission', 0.0)
        self.slippage = getattr(trade, 'slippage', 0.0)


class SwingChallengeTracker:
    """Track progress in €20 to €1M swing challenge"""
    def __init__(self):
        self.challenge_attempts = []
        self.current_attempt = {
            'start_time': datetime.now(),
            'start_balance': 20.0,
            'current_balance': 20.0,
            'trades': [],
            'peak_balance': 20.0,
            'max_drawdown': 0.0,
            'attempt_number': 1,
            'total_hold_time': 0,
            'avg_hold_time': 0,
            'longest_hold': 0,
            'shortest_hold': 999999
        }
    
    def start_new_attempt(self, attempt_number: int):
        """Start a new swing challenge attempt"""
        if self.current_attempt['trades']:
            self.challenge_attempts.append(self.current_attempt.copy())
        
        self.current_attempt = {
            'start_time': datetime.now(),
            'start_balance': 20.0,
            'current_balance': 20.0,
            'trades': [],
            'peak_balance': 20.0,
            'max_drawdown': 0.0,
            'attempt_number': attempt_number,
            'total_hold_time': 0,
            'avg_hold_time': 0,
            'longest_hold': 0,
            'shortest_hold': 999999
        }
    
    def update_balance(self, new_balance: float, trade_pnl: float, hold_time: int = 0):
        """Update current balance and swing-specific metrics"""
        self.current_attempt['current_balance'] = new_balance
        
        # Track peak balance
        if new_balance > self.current_attempt['peak_balance']:
            self.current_attempt['peak_balance'] = new_balance
        
        # Track max drawdown
        current_drawdown = self.current_attempt['peak_balance'] - new_balance
        if current_drawdown > self.current_attempt['max_drawdown']:
            self.current_attempt['max_drawdown'] = current_drawdown
        
        # Track hold time statistics
        if hold_time > 0:
            self.current_attempt['total_hold_time'] += hold_time
            trade_count = len([t for t in self.current_attempt['trades'] if t.get('type') == 'exit'])
            
            if trade_count > 0:
                self.current_attempt['avg_hold_time'] = self.current_attempt['total_hold_time'] / trade_count
            
            if hold_time > self.current_attempt['longest_hold']:
                self.current_attempt['longest_hold'] = hold_time
            
            if hold_time < self.current_attempt['shortest_hold']:
                self.current_attempt['shortest_hold'] = hold_time
    
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
        """Get comprehensive swing challenge progress summary"""
        current_level = self.get_current_level()
        next_target = self.get_next_target()
        
        # Calculate growth rate
        balance = self.current_attempt['current_balance']
        start_balance = self.current_attempt['start_balance']
        growth_rate = ((balance - start_balance) / start_balance) * 100
        
        # Calculate time in current attempt
        time_elapsed = datetime.now() - self.current_attempt['start_time']
        
        # Hold time statistics
        avg_hold_minutes = self.current_attempt['avg_hold_time'] / 60 if self.current_attempt['avg_hold_time'] > 0 else 0
        longest_hold_minutes = self.current_attempt['longest_hold'] / 60 if self.current_attempt['longest_hold'] > 0 else 0
        shortest_hold_minutes = self.current_attempt['shortest_hold'] / 60 if self.current_attempt['shortest_hold'] < 999999 else 0
        
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
            'total_attempts': len(self.challenge_attempts) + 1,
            'avg_hold_time_minutes': avg_hold_minutes,
            'longest_hold_minutes': longest_hold_minutes,
            'shortest_hold_minutes': shortest_hold_minutes,
            'total_hold_time_hours': self.current_attempt['total_hold_time'] / 3600
        }


class BTCSwingLogger:
    """
    Comprehensive BTC swing trade logger for €20 to €1M challenge
    Enhanced for swing trading with hold time analysis and percentage tracking
    """
    
    def __init__(self, log_file: str = "btc_swing_challenge.csv"):
        self.log_file = log_file
        self.trades = []
        
        # Challenge tracking
        self.challenge_tracker = SwingChallengeTracker()
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.session_start = datetime.now()
        
        # Swing-specific metrics
        self.largest_win_pct = 0.0
        self.largest_loss_pct = 0.0
        self.total_btc_traded = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_hold_time = 0
        self.hold_times = []
        
        # Setup files
        self._setup_csv_file()
        self._setup_challenge_log()
        
        logging.info(f"✅ BTC Swing Logger initialized - Challenge tracking active")
    
    def _setup_csv_file(self):
        """Setup CSV file with swing trading headers"""
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'price',
                    'order_id', 'status', 'trade_type', 'profit_loss', 'profit_loss_pct',
                    'commission', 'slippage', 'hold_time_seconds', 'hold_time_minutes',
                    'balance_after', 'challenge_level', 'attempt_number', 'swing_trade'
                ])
            logging.info(f"Created new swing challenge CSV: {self.log_file}")
        else:
            self._load_existing_trades()
    
    def _setup_challenge_log(self):
        """Setup swing challenge progress log"""
        self.challenge_log_file = self.log_file.replace('.csv', '_swing_challenge.json')
        
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
                logging.warning(f"Could not load swing challenge data: {e}")
    
    def _load_existing_trades(self):
        """Load existing swing trades from CSV"""
        try:
            with open(self.log_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Update performance metrics
                    if row.get('trade_type') == 'exit' and row.get('profit_loss'):
                        pnl = float(row['profit_loss'])
                        pnl_pct = float(row.get('profit_loss_pct', 0))
                        hold_time = int(row.get('hold_time_seconds', 0))
                        
                        self.total_trades += 1
                        self.total_pnl += pnl
                        self.total_hold_time += hold_time
                        self.hold_times.append(hold_time)
                        
                        if pnl > 0:
                            self.winning_trades += 1
                            self.largest_win_pct = max(self.largest_win_pct, pnl_pct)
                        else:
                            self.largest_loss_pct = min(self.largest_loss_pct, pnl_pct)
                    
                    # Track volume and costs
                    if row.get('quantity'):
                        self.total_btc_traded += float(row['quantity'])
                    if row.get('commission'):
                        self.total_commission += float(row['commission'])
                    if row.get('slippage'):
                        self.total_slippage += float(row['slippage'])
            
            logging.info(f"Loaded {self.total_trades} existing swing trades")
            
        except Exception as e:
            logging.error(f"Error loading swing trades: {e}")
    
    def log_trade(self, trade, trade_type: str = "entry", profit_loss: float = 0.0, 
                  profit_loss_pct: float = 0.0, hold_time: int = 0, current_balance: float = 0.0):
        """Log a BTC swing trade with enhanced metrics"""
        
        # Create trade record
        record = SwingTradeRecord(trade, trade_type, profit_loss, hold_time)
        self.trades.append(record)
        
        # Update challenge tracker
        if trade_type == "exit" and profit_loss != 0:
            self.challenge_tracker.update_balance(current_balance, profit_loss, hold_time)
            
            # Update performance metrics
            self.total_trades += 1
            self.total_pnl += profit_loss
            self.total_hold_time += hold_time
            self.hold_times.append(hold_time)
            
            if profit_loss > 0:
                self.winning_trades += 1
                self.largest_win_pct = max(self.largest_win_pct, profit_loss_pct)
            else:
                self.largest_loss_pct = min(self.largest_loss_pct, profit_loss_pct)
        
        # Track volume and costs
        self.total_btc_traded += record.quantity
        self.total_commission += record.commission
        self.total_slippage += record.slippage
        
        # Write to CSV
        self._write_to_csv(record, profit_loss_pct, hold_time, current_balance)
        
        # Save challenge progress
        self._save_challenge_progress()
        
        # Log the trade
        pnl_str = f" | P&L: €{profit_loss:+.2f} ({profit_loss_pct:+.2f}%)" if profit_loss != 0 else ""
        hold_str = f" | Hold: {hold_time//60}m{hold_time%60}s" if hold_time > 0 else ""
        btc_value = record.quantity * record.price
        
        logging.info(f"₿ SWING [{trade_type.upper()}]: {record.side.upper()} {record.quantity:.6f} BTC @ €{record.price:,.2f} (€{btc_value:,.2f}){pnl_str}{hold_str}")
        
        # Challenge progress update
        if trade_type == "exit":
            progress = self.challenge_tracker.get_progress_summary()
            logging.info(f"💰 Balance: €{current_balance:.2f} | Level: {progress['current_level']} | Target: €{progress['next_target']:.0f} | Avg Hold: {progress['avg_hold_time_minutes']:.1f}m")
    
    def _write_to_csv(self, record: SwingTradeRecord, profit_loss_pct: float, 
                     hold_time: int, current_balance: float):
        """Write swing trade record to CSV file"""
        
        try:
            progress = self.challenge_tracker.get_progress_summary()
            hold_time_minutes = hold_time / 60 if hold_time > 0 else 0
            
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
                    profit_loss_pct,
                    record.commission,
                    record.slippage,
                    hold_time,
                    hold_time_minutes,
                    current_balance,
                    progress['current_level'],
                    progress['attempt_number'],
                    True  # swing_trade flag
                ])
        except Exception as e:
            logging.error(f"Error writing swing trade to CSV: {e}")
    
    def _save_challenge_progress(self):
        """Save swing challenge progress to JSON file"""
        
        try:
            # Prepare data for JSON serialization
            current_attempt = self.challenge_tracker.current_attempt.copy()
            current_attempt['start_time'] = current_attempt['start_time'].isoformat()
            
            challenge_data = {
                'current_attempt': current_attempt,
                'attempts': self.challenge_tracker.challenge_attempts,
                'last_updated': datetime.now().isoformat(),
                'swing_trading_mode': True
            }
            
            with open(self.challenge_log_file, 'w') as f:
                json.dump(challenge_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving swing challenge progress: {e}")
    
    def start_new_challenge_attempt(self):
        """Start a new €20 to €1M swing challenge attempt"""
        
        current_attempt = self.challenge_tracker.current_attempt['attempt_number']
        new_attempt = current_attempt + 1
        
        logging.info(f"🔄 Starting new swing challenge attempt #{new_attempt}")
        
        # Log summary of previous attempt
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            avg_hold = (self.total_hold_time / self.total_trades) / 60  # minutes
            logging.info(f"   Previous attempt: {self.total_trades} trades, {win_rate:.1f}% win rate, €{self.total_pnl:+.2f} P&L, {avg_hold:.1f}m avg hold")
        
        # Reset tracking
        self.challenge_tracker.start_new_attempt(new_attempt)
        
        # Reset performance metrics for new attempt
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.largest_win_pct = 0.0
        self.largest_loss_pct = 0.0
        self.total_hold_time = 0
        self.hold_times = []
        
        self._save_challenge_progress()
        
        logging.info(f"✅ Swing challenge attempt #{new_attempt} started with €20")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive swing trading performance summary"""
        
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        avg_pnl = self.total_pnl / max(1, self.total_trades)
        session_duration = datetime.now() - self.session_start
        
        # Calculate profit factor
        wins = [t.profit_loss for t in self.trades if t.trade_type == 'exit' and t.profit_loss > 0]
        losses = [t.profit_loss for t in self.trades if t.trade_type == 'exit' and t.profit_loss < 0]
        
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Hold time statistics
        avg_hold_time = (self.total_hold_time / max(1, self.total_trades)) / 60  # minutes
        max_hold_time = max(self.hold_times) / 60 if self.hold_times else 0  # minutes
        min_hold_time = min(self.hold_times) / 60 if self.hold_times else 0  # minutes
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.total_trades - self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'average_pnl': avg_pnl,
            'largest_win_pct': self.largest_win_pct,
            'largest_loss_pct': self.largest_loss_pct,
            'profit_factor': profit_factor,
            'total_btc_traded': self.total_btc_traded,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'session_duration': str(session_duration).split('.')[0],
            'trades_logged': len(self.trades),
            'avg_hold_time_minutes': avg_hold_time,
            'max_hold_time_minutes': max_hold_time,
            'min_hold_time_minutes': min_hold_time,
            'total_hold_time_hours': self.total_hold_time / 3600,
            'swing_trading_mode': True
        }
    
    def get_challenge_summary(self) -> Dict:
        """Get €20 to €1M swing challenge summary"""
        
        progress = self.challenge_tracker.get_progress_summary()
        performance = self.get_performance_summary()
        
        # Calculate additional challenge metrics
        balance = progress['current_balance']
        time_to_million = "∞"
        
        if performance['average_pnl'] > 0 and self.total_trades > 0:
            # Estimate time to million based on current performance
            remaining = 1000000 - balance
            trades_needed = remaining / performance['average_pnl']
            # Assume average hold time + 30 minutes between trades
            time_per_trade = (performance['avg_hold_time_minutes'] + 30) / 60  # hours
            hours_needed = trades_needed * time_per_trade
            
            if hours_needed < 24:
                time_to_million = f"{hours_needed:.1f} hours"
            elif hours_needed < 168:  # 1 week
                time_to_million = f"{hours_needed/24:.1f} days"
            else:
                time_to_million = f"{hours_needed/168:.1f} weeks"
        
        # Calculate swing efficiency metrics
        trades_per_day = self.total_trades / max(1, (datetime.now() - self.session_start).days or 1)
        hold_efficiency = 100 - (performance['avg_hold_time_minutes'] / 5 * 100)  # 5 min = 0% efficiency
        
        return {
            **progress,
            **performance,
            'estimated_time_to_million': time_to_million,
            'success_probability': min(100, performance['win_rate'] * (performance['profit_factor'] / 2)),
            'risk_level': 'Low' if performance['profit_factor'] > 2 else 'Medium' if performance['profit_factor'] > 1.5 else 'High',
            'trades_per_day': trades_per_day,
            'hold_efficiency': max(0, hold_efficiency),
            'swing_trading_optimized': True
        }
    
    def get_hold_time_analysis(self) -> Dict:
        """Get detailed hold time analysis for swing optimization"""
        
        if not self.hold_times:
            return {'no_data': True}
        
        hold_times_minutes = [t / 60 for t in self.hold_times]
        
        # Categorize holds
        quick_holds = [t for t in hold_times_minutes if t < 2]    # < 2 minutes
        optimal_holds = [t for t in hold_times_minutes if 2 <= t <= 5]  # 2-5 minutes
        long_holds = [t for t in hold_times_minutes if t > 5]    # > 5 minutes
        
        # Performance by hold time category
        quick_trades = 0
        optimal_trades = 0
        long_trades = 0
        
        for i, trade in enumerate(self.trades):
            if trade.trade_type == 'exit' and i < len(self.hold_times):
                hold_minutes = self.hold_times[i] / 60
                if hold_minutes < 2:
                    quick_trades += 1
                elif 2 <= hold_minutes <= 5:
                    optimal_trades += 1
                else:
                    long_trades += 1
        
        return {
            'total_trades': len(self.hold_times),
            'avg_hold_minutes': sum(hold_times_minutes) / len(hold_times_minutes),
            'median_hold_minutes': sorted(hold_times_minutes)[len(hold_times_minutes)//2],
            'min_hold_minutes': min(hold_times_minutes),
            'max_hold_minutes': max(hold_times_minutes),
            'quick_holds_count': len(quick_holds),
            'optimal_holds_count': len(optimal_holds),
            'long_holds_count': len(long_holds),
            'quick_holds_pct': (len(quick_holds) / len(hold_times_minutes)) * 100,
            'optimal_holds_pct': (len(optimal_holds) / len(hold_times_minutes)) * 100,
            'long_holds_pct': (len(long_holds) / len(hold_times_minutes)) * 100,
            'swing_target_range': '2-5 minutes',
            'in_target_range_pct': (len(optimal_holds) / len(hold_times_minutes)) * 100
        }
    
    def print_challenge_status(self):
        """Print current swing challenge status"""
        
        summary = self.get_challenge_summary()
        hold_analysis = self.get_hold_time_analysis()
        
        print(f"\n₿ €20 → €1M SWING CHALLENGE STATUS")
        print("=" * 60)
        print(f"Attempt: #{summary['attempt_number']}")
        print(f"Current Balance: €{summary['current_balance']:,.2f}")
        print(f"Level: {summary['current_level']}")
        print(f"Next Target: €{summary['next_target']:,.0f}")
        print(f"Progress: {summary['progress_to_next']:.1f}%")
        print(f"Distance to €1M: €{summary['distance_to_million']:,.0f}")
        print(f"Growth Rate: {summary['growth_rate']:+.1f}%")
        print(f"Time Elapsed: {summary['time_elapsed']}")
        
        print(f"\n📊 SWING PERFORMANCE")
        print(f"Trades: {summary['total_trades']}")
        print(f"Win Rate: {summary['win_rate']:.1f}%")
        print(f"Total P&L: €{summary['total_pnl']:+.2f}")
        print(f"Profit Factor: {summary['profit_factor']:.2f}")
        print(f"Risk Level: {summary['risk_level']}")
        print(f"Trades/Day: {summary['trades_per_day']:.1f}")
        
        print(f"\n⏱️ HOLD TIME ANALYSIS")
        if not hold_analysis.get('no_data'):
            print(f"Average Hold: {hold_analysis['avg_hold_minutes']:.1f} minutes")
            print(f"Optimal Range (2-5m): {hold_analysis['in_target_range_pct']:.1f}%")
            print(f"Too Quick (<2m): {hold_analysis['quick_holds_pct']:.1f}%")
            print(f"Too Long (>5m): {hold_analysis['long_holds_pct']:.1f}%")
            print(f"Hold Efficiency: {summary['hold_efficiency']:.1f}%")
        
        if summary['estimated_time_to_million'] != "∞":
            print(f"\nEst. Time to €1M: {summary['estimated_time_to_million']}")
        
        print("=" * 60)
    
    def export_challenge_data(self, filename: Optional[str] = None) -> str:
        """Export all swing challenge data to JSON file"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"btc_swing_challenge_export_{timestamp}.json"
        
        try:
            export_data = {
                'challenge_summary': self.get_challenge_summary(),
                'performance_summary': self.get_performance_summary(),
                'hold_time_analysis': self.get_hold_time_analysis(),
                'all_attempts': self.challenge_tracker.challenge_attempts,
                'current_attempt': self.challenge_tracker.current_attempt,
                'recent_trades': [
                    {
                        'timestamp': t.timestamp,
                        'side': t.side,
                        'quantity': t.quantity,
                        'price': t.price,
                        'type': t.trade_type,
                        'pnl': t.profit_loss,
                        'hold_time': t.hold_time
                    }
                    for t in self.trades[-20:]  # Last 20 trades
                ],
                'trading_mode': 'swing_scalping',
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logging.info(f"✅ Swing challenge data exported to: {filename}")
            return filename
            
        except Exception as e:
            logging.error(f"Error exporting swing challenge data: {e}")
            return ""
    
    def get_swing_insights(self) -> Dict:
        """Get insights for swing trading optimization"""
        
        hold_analysis = self.get_hold_time_analysis()
        performance = self.get_performance_summary()
        
        insights = {
            'trading_mode': 'swing_scalping',
            'optimization_suggestions': [],
            'performance_rating': 'Unknown',
            'hold_time_optimization': 'Unknown'
        }
        
        if not hold_analysis.get('no_data'):
            # Hold time insights
            if hold_analysis['avg_hold_minutes'] < 2:
                insights['optimization_suggestions'].append("Consider holding positions longer for better swing profits")
                insights['hold_time_optimization'] = 'Hold Longer'
            elif hold_analysis['avg_hold_minutes'] > 5:
                insights['optimization_suggestions'].append("Consider shorter holds to increase trade frequency")
                insights['hold_time_optimization'] = 'Hold Shorter'
            else:
                insights['hold_time_optimization'] = 'Optimal'
            
            # Performance insights
            if performance['win_rate'] > 70:
                insights['performance_rating'] = 'Excellent'
            elif performance['win_rate'] > 60:
                insights['performance_rating'] = 'Good'
            elif performance['win_rate'] > 50:
                insights['performance_rating'] = 'Average'
            else:
                insights['performance_rating'] = 'Needs Improvement'
                insights['optimization_suggestions'].append("Focus on higher confidence setups")
            
            # Profit factor insights
            if performance['profit_factor'] < 1.5:
                insights['optimization_suggestions'].append("Improve risk/reward ratio - aim for larger profits vs losses")
            
            # Hold efficiency insights
            if hold_analysis['in_target_range_pct'] < 60:
                insights['optimization_suggestions'].append("Aim for 2-5 minute holds for optimal swing trading")
        
        return insights
    
    def cleanup(self):
        """Clean up and save final swing challenge state"""
        
        try:
            # Save final challenge progress
            self._save_challenge_progress()
            
            # Print final summaries
            self.print_challenge_status()
            
            # Print swing insights
            insights = self.get_swing_insights()
            if insights['optimization_suggestions']:
                print(f"\n💡 SWING TRADING INSIGHTS:")
                for suggestion in insights['optimization_suggestions']:
                    print(f"   • {suggestion}")
                print(f"   Performance Rating: {insights['performance_rating']}")
                print(f"   Hold Time Optimization: {insights['hold_time_optimization']}")
            
            # Export final data
            export_file = self.export_challenge_data()
            if export_file:
                print(f"\n💾 Swing challenge data exported to: {export_file}")
            
            logging.info("✅ BTC Swing Logger cleanup completed")
            
        except Exception as e:
            logging.error(f"Error during swing logger cleanup: {e}")


if __name__ == "__main__":
    # Test BTC swing logger
    logger = BTCSwingLogger()
    
    # Mock trade object for swing trading
    class MockSwingTrade:
        def __init__(self, side: str, price: float):
            self.timestamp = datetime.now().isoformat()
            self.symbol = 'BTCUSD'
            self.side = side
            self.quantity = 0.0005  # Typical swing size
            self.price = price
            self.fill_price = price + (2 if side == 'buy' else -2)  # Swing slippage
            self.order_id = f'swing_{side}_{int(datetime.now().timestamp())}'
            self.status = type('Status', (), {'value': 'filled'})()
            self.commission = 1.00
            self.slippage = 2.00
    
    print("🧪 Testing BTC Swing Logger...")
    
    # Simulate swing trading sequence
    entry_trade = MockSwingTrade('buy', 43250.00)
    logger.log_trade(entry_trade, 'entry', current_balance=20.0)
    
    # Simulate hold time (3 minutes = 180 seconds)
    hold_time = 180
    exit_trade = MockSwingTrade('sell', 43358.00)  # 2.5% profit
    profit_pnl = 14.50  # Example profit
    profit_pct = 2.5
    logger.log_trade(exit_trade, 'exit', profit_loss=profit_pnl, 
                    profit_loss_pct=profit_pct, hold_time=hold_time, current_balance=34.50)
    
    # Simulate another trade
    entry_trade2 = MockSwingTrade('sell', 43350.00)  # Short entry
    logger.log_trade(entry_trade2, 'entry', current_balance=34.50)
    
    hold_time2 = 240  # 4 minutes
    exit_trade2 = MockSwingTrade('buy', 43220.00)  # Profitable short
    profit_pnl2 = 12.20
    profit_pct2 = 1.8
    logger.log_trade(exit_trade2, 'exit', profit_loss=profit_pnl2,
                    profit_loss_pct=profit_pct2, hold_time=hold_time2, current_balance=46.70)
    
    # Print status
    logger.print_challenge_status()
    
    # Test hold time analysis
    hold_analysis = logger.get_hold_time_analysis()
    print(f"\n⏱️ Hold Time Analysis:")
    for key, value in hold_analysis.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.2f}")
    
    # Test swing insights
    insights = logger.get_swing_insights()
    print(f"\n💡 Swing Insights:")
    print(f"   Performance Rating: {insights['performance_rating']}")
    print(f"   Hold Time Optimization: {insights['hold_time_optimization']}")
    for suggestion in insights['optimization_suggestions']:
        print(f"   • {suggestion}")
    
    # Test export
    export_file = logger.export_challenge_data()
    print(f"\nExported to: {export_file}")
    
    # Cleanup
    logger.cleanup()
    
    print("✅ BTC Swing Logger test completed")