#!/usr/bin/env python3
"""
Simple Trade Logger for BTC/USD
"""

import csv
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional


class SimpleBTCTradeRecord:
    """Simple BTC trade record"""
    def __init__(self, trade, trade_type: str = "entry", profit_loss: float = 0.0):
        self.timestamp = trade.timestamp
        self.symbol = trade.symbol
        self.side = trade.side
        self.quantity = trade.quantity
        self.price = trade.price
        self.trade_id = trade.trade_id
        self.status = trade.status
        self.trade_type = trade_type  # 'entry' or 'exit'
        self.profit_loss = profit_loss


class SimpleBTCTradeLogger:
    """Simple trade logger for BTC/USD"""
    
    def __init__(self, log_file: str = "btcusd_trades.csv"):
        self.log_file = log_file
        self.trades = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.session_start = datetime.now()
        
        # BTC-specific metrics
        self.largest_win_usd = 0.0
        self.largest_loss_usd = 0.0
        self.total_btc_traded = 0.0
        self.avg_hold_time = 0.0
        
        # Setup CSV file
        self._setup_csv_file()
        
        logging.info(f"✅ BTC trade logger initialized - File: {log_file}")
    
    def _setup_csv_file(self):
        """Setup CSV file with headers"""
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'price', 
                    'trade_id', 'status', 'trade_type', 'profit_loss', 'btc_value'
                ])
            logging.info(f"Created new BTC CSV file: {self.log_file}")
        else:
            # Load existing trades
            self._load_existing_trades()
    
    def _load_existing_trades(self):
        """Load existing trades from CSV"""
        
        try:
            with open(self.log_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    record = SimpleBTCTradeRecord(
                        trade=type('Trade', (), {
                            'timestamp': row['timestamp'],
                            'symbol': row['symbol'],
                            'side': row['side'],
                            'quantity': float(row['quantity']),
                            'price': float(row['price']),
                            'trade_id': row['trade_id'],
                            'status': row['status']
                        })(),
                        trade_type=row.get('trade_type', 'entry'),
                        profit_loss=float(row.get('profit_loss', 0))
                    )
                    self.trades.append(record)
                    
                    # Update performance metrics
                    if record.trade_type == 'exit' and record.profit_loss != 0:
                        self.total_trades += 1
                        self.total_pnl += record.profit_loss
                        if record.profit_loss > 0:
                            self.winning_trades += 1
                            self.largest_win_usd = max(self.largest_win_usd, record.profit_loss)
                        else:
                            self.largest_loss_usd = min(self.largest_loss_usd, record.profit_loss)
                    
                    # Track BTC volume
                    self.total_btc_traded += record.quantity
            
            logging.info(f"Loaded {len(self.trades)} existing BTC trades")
            
        except Exception as e:
            logging.error(f"Error loading BTC trades: {e}")
    
    def log_trade(self, trade, trade_type: str = "entry", profit_loss: float = 0.0):
        """Log a BTC trade to CSV and memory"""
        
        # Create trade record
        record = SimpleBTCTradeRecord(trade, trade_type, profit_loss)
        self.trades.append(record)
        
        # Write to CSV
        self._write_to_csv(record)
        
        # Update performance metrics for exit trades
        if trade_type == "exit" and profit_loss != 0:
            self.total_trades += 1
            self.total_pnl += profit_loss
            
            if profit_loss > 0:
                self.winning_trades += 1
                self.largest_win_usd = max(self.largest_win_usd, profit_loss)
            else:
                self.largest_loss_usd = min(self.largest_loss_usd, profit_loss)
        
        # Track BTC volume
        self.total_btc_traded += trade.quantity
        
        # Log the trade with BTC-specific format
        btc_value = trade.quantity * trade.price
        pnl_str = f" | P&L: ${profit_loss:+.2f}" if profit_loss != 0 else ""
        logging.info(f"₿ TRADE [{trade_type.upper()}]: {trade.side.upper()} {trade.quantity} BTC @ ${trade.price:,.2f} (${btc_value:,.2f}){pnl_str}")
    
    def _write_to_csv(self, record: SimpleBTCTradeRecord):
        """Write trade record to CSV file"""
        
        try:
            btc_value = record.quantity * record.price
            
            with open(self.log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    record.timestamp,
                    record.symbol,
                    record.side,
                    record.quantity,
                    record.price,
                    record.trade_id,
                    record.status,
                    record.trade_type,
                    record.profit_loss,
                    btc_value
                ])
        except Exception as e:
            logging.error(f"Error writing BTC trade to CSV: {e}")
    
    def calculate_trade_pnl(self, entry_trade, exit_trade) -> float:
        """Calculate P&L between entry and exit trades"""
        
        try:
            entry_price = entry_trade.price
            exit_price = exit_trade.price
            quantity = entry_trade.quantity
            side = entry_trade.side
            
            if side == 'buy':
                pnl = (exit_price - entry_price) * quantity
            else:  # sell
                pnl = (entry_price - exit_price) * quantity
            
            return round(pnl, 2)
            
        except Exception as e:
            logging.error(f"Error calculating BTC P&L: {e}")
            return 0.0
    
    def get_performance_summary(self) -> Dict:
        """Get BTC performance summary"""
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_pnl = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
        session_duration = datetime.now() - self.session_start
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.total_trades - self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'average_pnl': avg_pnl,
            'largest_win': self.largest_win_usd,
            'largest_loss': self.largest_loss_usd,
            'total_btc_traded': self.total_btc_traded,
            'session_duration': str(session_duration).split('.')[0],  # Remove microseconds
            'trades_logged': len(self.trades)
        }
    
    def get_btc_specific_metrics(self) -> Dict:
        """Get BTC-specific trading metrics"""
        
        if self.total_trades == 0:
            return {}
        
        # Calculate average trade size
        avg_trade_size = self.total_btc_traded / len(self.trades) if self.trades else 0
        
        # Calculate profit factor
        wins = [t.profit_loss for t in self.trades if t.trade_type == 'exit' and t.profit_loss > 0]
        losses = [t.profit_loss for t in self.trades if t.trade_type == 'exit' and t.profit_loss < 0]
        
        profit_factor = abs(sum(wins) / sum(losses)) if losses else float('inf')
        
        # Calculate return on BTC traded
        total_btc_value = sum(t.quantity * t.price for t in self.trades if t.trade_type == 'entry')
        roi_percentage = (self.total_pnl / total_btc_value * 100) if total_btc_value > 0 else 0
        
        return {
            'avg_trade_size_btc': avg_trade_size,
            'total_btc_volume': self.total_btc_traded,
            'profit_factor': profit_factor,
            'roi_percentage': roi_percentage,
            'avg_win': sum(wins) / len(wins) if wins else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'largest_win_usd': self.largest_win_usd,
            'largest_loss_usd': self.largest_loss_usd
        }
    
    def get_recent_trades(self, count: int = 10) -> List[SimpleBTCTradeRecord]:
        """Get recent BTC trades"""
        return self.trades[-count:] if self.trades else []
    
    def get_daily_summary(self) -> Dict:
        """Get today's BTC trading summary"""
        
        today = datetime.now().date()
        today_trades = []
        
        for trade in self.trades:
            trade_date = datetime.fromisoformat(trade.timestamp).date()
            if trade_date == today:
                today_trades.append(trade)
        
        # Calculate today's stats
        today_exits = [t for t in today_trades if t.trade_type == 'exit']
        today_pnl = sum(t.profit_loss for t in today_exits)
        today_wins = sum(1 for t in today_exits if t.profit_loss > 0)
        today_total = len(today_exits)
        today_btc_volume = sum(t.quantity for t in today_trades)
        
        return {
            'date': today.isoformat(),
            'total_trades': today_total,
            'winning_trades': today_wins,
            'total_pnl': today_pnl,
            'win_rate': (today_wins / today_total * 100) if today_total > 0 else 0,
            'btc_volume': today_btc_volume,
            'trades_logged': len(today_trades)
        }
    
    def print_performance_summary(self):
        """Print BTC performance summary to console"""
        
        summary = self.get_performance_summary()
        btc_metrics = self.get_btc_specific_metrics()
        
        print("\n" + "="*60)
        print("           ₿ BTC TRADING PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Session Duration: {summary['session_duration']}")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Winning Trades: {summary['winning_trades']}")
        print(f"Losing Trades: {summary['losing_trades']}")
        print(f"Win Rate: {summary['win_rate']:.1f}%")
        print(f"Total P&L: ${summary['total_pnl']:+.2f}")
        print(f"Average P&L: ${summary['average_pnl']:+.2f}")
        print(f"Largest Win: ${summary['largest_win']:+.2f}")
        print(f"Largest Loss: ${summary['largest_loss']:+.2f}")
        print(f"Total BTC Traded: {summary['total_btc_traded']:.6f} BTC")
        
        if btc_metrics:
            print(f"ROI: {btc_metrics['roi_percentage']:+.2f}%")
            print(f"Profit Factor: {btc_metrics['profit_factor']:.2f}")
            print(f"Avg Trade Size: {btc_metrics['avg_trade_size_btc']:.6f} BTC")
        
        print(f"Trades Logged: {summary['trades_logged']}")
        print("="*60)
    
    def print_daily_summary(self):
        """Print today's BTC summary"""
        
        daily = self.get_daily_summary()
        
        print(f"\n₿ TODAY'S BTC SUMMARY ({daily['date']}):")
        print(f"   Trades: {daily['total_trades']}")
        print(f"   Wins: {daily['winning_trades']}")
        print(f"   Win Rate: {daily['win_rate']:.1f}%")
        print(f"   P&L: ${daily['total_pnl']:+.2f}")
        print(f"   BTC Volume: {daily['btc_volume']:.6f} BTC")
    
    def export_trades(self, filename: Optional[str] = None) -> str:
        """Export BTC trades to a new CSV file"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"btc_trades_export_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'price',
                    'trade_id', 'status', 'trade_type', 'profit_loss', 'btc_value'
                ])
                
                for record in self.trades:
                    btc_value = record.quantity * record.price
                    writer.writerow([
                        record.timestamp,
                        record.symbol,
                        record.side,
                        record.quantity,
                        record.price,
                        record.trade_id,
                        record.status,
                        record.trade_type,
                        record.profit_loss,
                        btc_value
                    ])
            
            logging.info(f"✅ BTC trades exported to: {filename}")
            return filename
            
        except Exception as e:
            logging.error(f"Error exporting BTC trades: {e}")
            return ""
    
    def cleanup(self):
        """Clean up BTC logger resources"""
        
        try:
            # Print final summaries
            self.print_performance_summary()
            self.print_daily_summary()
            
            # Print BTC-specific statistics
            btc_metrics = self.get_btc_specific_metrics()
            if btc_metrics:
                print(f"\n₿ BTC SPECIFIC METRICS:")
                print(f"   Average Win: ${btc_metrics['avg_win']:+.2f}")
                print(f"   Average Loss: ${btc_metrics['avg_loss']:+.2f}")
                print(f"   Profit Factor: {btc_metrics['profit_factor']:.2f}")
                print(f"   ROI: {btc_metrics['roi_percentage']:+.2f}%")
                print(f"   Total BTC Volume: {btc_metrics['total_btc_volume']:.6f} BTC")
            
            logging.info("✅ BTC trade logger cleaned up successfully")
            
        except Exception as e:
            logging.error(f"Error during BTC logger cleanup: {e}")


if __name__ == "__main__":
    # Test BTC trade logger
    logger = SimpleBTCTradeLogger()
    
    # Mock trade object for testing
    class MockBTCTrade:
        def __init__(self, side: str, price: float):
            self.timestamp = datetime.now().isoformat()
            self.symbol = 'BTCUSD'
            self.side = side
            self.quantity = 0.001
            self.price = price
            self.trade_id = f'btc_{side}_{int(datetime.now().timestamp())}'
            self.status = 'filled'
    
    # Test multiple BTC trades
    print("🧪 Testing BTC trade logger...")
    
    # Trade 1: Profitable
    entry1 = MockBTCTrade('buy', 43250.00)
    logger.log_trade(entry1, trade_type="entry")
    
    exit1 = MockBTCTrade('sell', 43350.00)
    pnl1 = logger.calculate_trade_pnl(entry1, exit1)
    logger.log_trade(exit1, trade_type="exit", profit_loss=pnl1)
    
    # Trade 2: Loss
    entry2 = MockBTCTrade('buy', 43300.00)
    logger.log_trade(entry2, trade_type="entry")
    
    exit2 = MockBTCTrade('sell', 43250.00)
    pnl2 = logger.calculate_trade_pnl(entry2, exit2)
    logger.log_trade(exit2, trade_type="exit", profit_loss=pnl2)
    
    # Print summaries
    logger.print_performance_summary()
    logger.print_daily_summary()
    
    # Test BTC metrics
    btc_metrics = logger.get_btc_specific_metrics()
    print(f"BTC Metrics: {btc_metrics}")
    
    # Test export
    export_file = logger.export_trades()
    print(f"Exported to: {export_file}")
    
    # Cleanup
    logger.cleanup()