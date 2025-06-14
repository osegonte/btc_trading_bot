#!/usr/bin/env python3
"""
Enhanced Trade Execution for BTC/USD with Risk Management
Includes: Position tracking, wash trade protection, enhanced error handling
"""

import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BTCTrade:
    """Enhanced BTC trade data structure"""
    symbol: str
    side: str
    quantity: float
    price: float
    status: OrderStatus = OrderStatus.PENDING
    trade_id: str = ""
    timestamp: str = ""
    fill_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    
    def __post_init__(self):
        if not self.trade_id:
            self.trade_id = f"{self.symbol}_{self.side}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class PositionTracker:
    """Track positions to prevent conflicts"""
    def __init__(self):
        self.current_position: Optional[BTCTrade] = None
        self.position_history: List[BTCTrade] = []
        self.daily_trades = 0
        self.wash_trade_window = timedelta(minutes=5)
    
    def can_open_position(self, side: str, price: float) -> tuple[bool, str]:
        """Check if position can be opened"""
        if self.current_position:
            return False, f"Already have {self.current_position.side} position"
        
        # Check for wash trade risk
        if self._is_wash_trade_risk(side, price):
            return False, "Potential wash trade detected"
        
        return True, "OK"
    
    def _is_wash_trade_risk(self, side: str, price: float) -> bool:
        """Check for wash trade patterns"""
        if not self.position_history:
            return False
        
        cutoff_time = datetime.now() - self.wash_trade_window
        recent_trades = [
            trade for trade in self.position_history[-10:]
            if datetime.fromisoformat(trade.timestamp) > cutoff_time
        ]
        
        # Count similar trades
        similar_trades = 0
        for trade in recent_trades:
            if (trade.side == side and 
                abs(trade.price - price) / price < 0.01):  # Within 1%
                similar_trades += 1
        
        return similar_trades >= 3
    
    def open_position(self, trade: BTCTrade):
        """Record position opening"""
        self.current_position = trade
        self.position_history.append(trade)
        self.daily_trades += 1
    
    def close_position(self) -> Optional[BTCTrade]:
        """Record position closing"""
        if self.current_position:
            closed_position = self.current_position
            self.current_position = None
            return closed_position
        return None


class EnhancedBTCTradeExecutor:
    """Enhanced trade executor for BTC/USD with comprehensive risk management"""
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Configuration
        self.paper_trading = config.get('paper_trading', True)
        self.api_key = config.get('api_key', "")
        self.secret_key = config.get('secret_key', "")
        self.api = None
        
        # Enhanced settings
        self.min_btc_quantity = 0.0001
        self.max_slippage = 3.0  # Max $3 slippage for BTC
        self.order_timeout = 10.0  # 10 seconds timeout
        self.max_retries = 3
        
        # Account simulation
        self.account_balance = config.get('initial_balance', 100000.0)
        self.cash = self.account_balance
        self.btc_holdings = 0.0
        
        # Position tracking
        self.position_tracker = PositionTracker()
        
        # Performance tracking
        self.total_orders = 0
        self.successful_orders = 0
        self.total_slippage = 0.0
        self.total_commission = 0.0
        
        # Risk management
        self.daily_loss_limit = config.get('daily_loss_limit', 500.0)
        self.daily_pnl = 0.0
        self.max_position_size = config.get('max_position_size', 0.01)
        
        # Connection management
        self.connection_stable = True
        self.last_order_time = None
        self.min_order_interval = 2.0  # Minimum 2 seconds between orders
        
        # Initialize API if credentials provided
        if self.api_key and self.secret_key:
            self._init_alpaca_api()
        else:
            logging.info("🎮 Enhanced BTC executor in simulation mode")
    
    def _init_alpaca_api(self):
        """Initialize Alpaca API with enhanced error handling"""
        try:
            import alpaca_trade_api as tradeapi
            
            base_url = 'https://paper-api.alpaca.markets' if self.paper_trading else 'https://api.alpaca.markets'
            
            self.api = tradeapi.REST(
                self.api_key, 
                self.secret_key, 
                base_url, 
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            self.account_balance = float(account.portfolio_value)
            self.cash = float(account.cash)
            
            mode = "📄 PAPER" if self.paper_trading else "🔴 LIVE"
            logging.info(f"✅ Enhanced Alpaca connected - {mode} mode")
            logging.info(f"💰 Account: ${self.account_balance:,.2f}")
            
            self.connection_stable = True
            
        except ImportError:
            logging.warning("alpaca-trade-api not installed - using enhanced simulation")
            self.api = None
            self.connection_stable = False
        except Exception as e:
            logging.error(f"Enhanced Alpaca connection failed: {e}")
            self.api = None
            self.connection_stable = False
    
    def place_order(self, symbol: str, side: str, quantity: float, current_price: float) -> BTCTrade:
        """Place enhanced BTC order with comprehensive validation"""
        
        # Create trade object
        trade = BTCTrade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=current_price
        )
        
        self.total_orders += 1
        
        # Enhanced pre-order validation
        validation_result = self._validate_order(trade)
        if not validation_result[0]:
            trade.status = OrderStatus.REJECTED
            logging.warning(f"Order rejected: {validation_result[1]}")
            return trade
        
        # Execute order
        try:
            if self.api and self.connection_stable:
                return self._execute_alpaca_order(trade)
            else:
                return self._execute_simulated_order(trade)
                
        except Exception as e:
            logging.error(f"Enhanced order execution failed: {e}")
            trade.status = OrderStatus.FAILED
            return trade
    
    def _validate_order(self, trade: BTCTrade) -> tuple[bool, str]:
        """Comprehensive order validation"""
        
        # Basic validation
        if trade.quantity < self.min_btc_quantity:
            return False, f"Order too small: {trade.quantity} < {self.min_btc_quantity}"
        
        if trade.quantity > self.max_position_size:
            return False, f"Order too large: {trade.quantity} > {self.max_position_size}"
        
        if trade.price <= 0:
            return False, f"Invalid price: ${trade.price}"
        
        # Position validation
        can_open, reason = self.position_tracker.can_open_position(trade.side, trade.price)
        if not can_open:
            return False, reason
        
        # Timing validation
        if self.last_order_time:
            time_since_last = time.time() - self.last_order_time
            if time_since_last < self.min_order_interval:
                return False, f"Too soon since last order: {time_since_last:.1f}s"
        
        # Balance validation
        trade_value = trade.quantity * trade.price
        if trade.side == 'buy':
            if self.cash < trade_value:
                return False, f"Insufficient cash: ${trade_value:,.2f} needed, ${self.cash:,.2f} available"
        else:  # sell
            if self.btc_holdings < trade.quantity:
                return False, f"Insufficient BTC: {trade.quantity} needed, {self.btc_holdings} available"
        
        # Daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            return False, f"Daily loss limit reached: ${self.daily_pnl:.2f}"
        
        return True, "OK"
    
    def _execute_alpaca_order(self, trade: BTCTrade) -> BTCTrade:
        """Execute order via Alpaca API with retries"""
        
        for attempt in range(self.max_retries):
            try:
                # Submit order
                order = self.api.submit_order(
                    symbol=trade.symbol,
                    qty=trade.quantity,
                    side=trade.side,
                    type='market',
                    time_in_force='gtc'
                )
                
                # Wait for fill with timeout
                start_time = time.time()
                while time.time() - start_time < self.order_timeout:
                    updated_order = self.api.get_order(order.id)
                    
                    if updated_order.status == 'filled':
                        # Order filled successfully
                        fill_price = float(updated_order.filled_avg_price)
                        trade.fill_price = fill_price
                        trade.status = OrderStatus.FILLED
                        trade.slippage = abs(fill_price - trade.price)
                        
                        # Update tracking
                        self._update_holdings(trade, fill_price)
                        self.position_tracker.open_position(trade)
                        self.successful_orders += 1
                        self.total_slippage += trade.slippage
                        self.last_order_time = time.time()
                        
                        logging.info(f"✅ Alpaca BTC order filled: {trade.side.upper()} {trade.quantity} @ ${fill_price:,.2f}")
                        return trade
                    
                    elif updated_order.status in ['rejected', 'canceled']:
                        trade.status = OrderStatus.REJECTED
                        logging.warning(f"❌ Alpaca order {updated_order.status}: {order.id}")
                        return trade
                    
                    time.sleep(0.1)
                
                # Timeout - order still pending
                trade.status = OrderStatus.PENDING
                logging.warning(f"⚠️ Alpaca order timeout: {order.id}")
                return trade
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle specific Alpaca errors
                if 'wash trade' in error_msg:
                    trade.status = OrderStatus.REJECTED
                    logging.warning(f"❌ Wash trade detected: {e}")
                    return trade
                elif 'insufficient' in error_msg:
                    trade.status = OrderStatus.REJECTED
                    logging.warning(f"❌ Insufficient balance: {e}")
                    return trade
                elif attempt < self.max_retries - 1:
                    logging.warning(f"⚠️ Alpaca order attempt {attempt + 1} failed: {e}")
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    trade.status = OrderStatus.FAILED
                    logging.error(f"❌ Alpaca order failed after {self.max_retries} attempts: {e}")
                    return trade
        
        return trade
    
    def _execute_simulated_order(self, trade: BTCTrade) -> BTCTrade:
        """Execute simulated order with realistic behavior"""
        
        try:
            # Simulate order processing time
            time.sleep(0.1 + np.random.uniform(0, 0.2))
            
            # Calculate realistic slippage
            slippage_direction = 1 if trade.side == 'buy' else -1
            base_slippage = np.random.uniform(0.5, self.max_slippage)
            
            # Higher slippage during high volatility (simulated)
            volatility_factor = 1 + np.random.uniform(0, 0.5)
            slippage = base_slippage * volatility_factor * slippage_direction
            
            fill_price = max(1.0, trade.price + slippage)
            
            # Simulate occasional order rejection (1% chance)
            if np.random.random() < 0.01:
                trade.status = OrderStatus.REJECTED
                logging.info(f"🎮 Simulated order rejection")
                return trade
            
            # Order filled
            trade.fill_price = fill_price
            trade.status = OrderStatus.FILLED
            trade.slippage = abs(slippage)
            trade.commission = trade.quantity * fill_price * 0.001  # 0.1% commission
            
            # Update tracking
            self._update_holdings(trade, fill_price)
            self.position_tracker.open_position(trade)
            self.successful_orders += 1
            self.total_slippage += trade.slippage
            self.total_commission += trade.commission
            self.last_order_time = time.time()
            
            # Detailed logging
            trade_value = trade.quantity * fill_price
            slippage_str = f"(slippage: ${slippage:+.2f})" if abs(slippage) > 0.1 else ""
            
            logging.info(f"✅ Simulated BTC order: {trade.side.upper()} {trade.quantity} BTC @ ${fill_price:,.2f} ${slippage_str}")
            
            return trade
            
        except Exception as e:
            logging.error(f"Simulated order error: {e}")
            trade.status = OrderStatus.FAILED
            return trade
    
    def _update_holdings(self, trade: BTCTrade, fill_price: float):
        """Update account holdings"""
        trade_value = trade.quantity * fill_price
        
        if trade.side == 'buy':
            self.cash -= trade_value + trade.commission
            self.btc_holdings += trade.quantity
        else:  # sell
            self.cash += trade_value - trade.commission
            self.btc_holdings -= trade.quantity
        
        # Ensure no negative holdings
        self.btc_holdings = max(0, self.btc_holdings)
        self.cash = max(0, self.cash)
    
    def close_position(self, current_price: float) -> Optional[BTCTrade]:
        """Close current position if exists"""
        open_position = self.position_tracker.close_position()
        if not open_position:
            return None
        
        # Determine close side
        close_side = 'sell' if open_position.side == 'buy' else 'buy'
        
        # Place closing order
        close_trade = self.place_order(
            open_position.symbol,
            close_side,
            open_position.quantity,
            current_price
        )
        
        if close_trade.status == OrderStatus.FILLED:
            # Calculate P&L
            if open_position.side == 'buy':
                pnl = (close_trade.fill_price - open_position.fill_price) * open_position.quantity
            else:
                pnl = (open_position.fill_price - close_trade.fill_price) * open_position.quantity
            
            # Account for commissions
            pnl -= (open_position.commission + close_trade.commission)
            
            self.daily_pnl += pnl
            
            logging.info(f"Position closed: P&L ${pnl:+.2f}")
        
        return close_trade
    
    def get_account_info(self) -> Dict:
        """Get comprehensive account information"""
        
        if self.api and self.connection_stable:
            try:
                account = self.api.get_account()
                return {
                    'balance': float(account.portfolio_value),
                    'cash': float(account.cash),
                    'buying_power': float(account.buying_power),
                    'day_trade_count': int(account.daytrade_count),
                    'account_blocked': account.account_blocked
                }
            except Exception as e:
                logging.warning(f"Failed to get live account info: {e}")
        
        # Calculate portfolio value
        btc_value = self.btc_holdings * 43000  # Approximate BTC value
        total_value = self.cash + btc_value
        
        return {
            'balance': total_value,
            'cash': self.cash,
            'buying_power': self.cash,
            'btc_holdings': self.btc_holdings,
            'btc_value': btc_value,
            'daily_pnl': self.daily_pnl,
            'connection_stable': self.connection_stable
        }
    
    def get_position_info(self) -> Dict:
        """Get current position information"""
        
        if self.api and self.connection_stable:
            try:
                positions = self.api.list_positions()
                for pos in positions:
                    if pos.symbol == 'BTCUSD':
                        return {
                            'has_position': True,
                            'quantity': float(pos.qty),
                            'side': 'long' if float(pos.qty) > 0 else 'short',
                            'avg_price': float(pos.avg_cost),
                            'market_value': float(pos.market_value),
                            'unrealized_pnl': float(pos.unrealized_pnl)
                        }
            except Exception as e:
                logging.warning(f"Failed to get live position info: {e}")
        
        # Return simulated position info
        current_pos = self.position_tracker.current_position
        if current_pos:
            return {
                'has_position': True,
                'quantity': current_pos.quantity,
                'side': 'long' if current_pos.side == 'buy' else 'short',
                'avg_price': current_pos.fill_price or current_pos.price,
                'market_value': current_pos.quantity * (current_pos.fill_price or current_pos.price),
                'trade_id': current_pos.trade_id
            }
        
        return {'has_position': False}
    
    def get_trading_stats(self) -> Dict:
        """Get comprehensive trading statistics"""
        
        success_rate = (self.successful_orders / max(1, self.total_orders)) * 100
        avg_slippage = self.total_slippage / max(1, self.successful_orders)
        avg_commission = self.total_commission / max(1, self.successful_orders)
        
        return {
            'total_orders': self.total_orders,
            'successful_orders': self.successful_orders,
            'success_rate': success_rate,
            'avg_slippage': avg_slippage,
            'total_slippage': self.total_slippage,
            'avg_commission': avg_commission,
            'total_commission': self.total_commission,
            'daily_trades': self.position_tracker.daily_trades,
            'daily_pnl': self.daily_pnl,
            'connection_stable': self.connection_stable,
            'mode': 'Paper' if self.paper_trading else 'Live'
        }
    
    def get_risk_metrics(self) -> Dict:
        """Get risk management metrics"""
        
        daily_risk_utilization = (abs(self.daily_pnl) / self.daily_loss_limit) * 100 if self.daily_pnl < 0 else 0
        
        return {
            'daily_pnl': self.daily_pnl,
            'daily_loss_limit': self.daily_loss_limit,
            'risk_utilization_pct': daily_risk_utilization,
            'max_position_size': self.max_position_size,
            'min_order_interval': self.min_order_interval,
            'wash_trade_protection': True,
            'circuit_breaker_active': daily_risk_utilization >= 100
        }
    
    def is_market_open(self) -> bool:
        """Check if crypto market is open (always true for BTC)"""
        
        if self.api:
            try:
                # For crypto, check if we can get market data
                return True  # Crypto markets are 24/7
            except:
                pass
        
        return True  # Crypto never closes
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0.0
        self.position_tracker.daily_trades = 0
        logging.info("📊 Daily execution stats reset")


if __name__ == "__main__":
    # Test enhanced trade executor
    import numpy as np
    
    config = {
        'paper_trading': True,
        'initial_balance': 10000,
        'max_position_size': 0.01,
        'daily_loss_limit': 500
    }
    
    executor = EnhancedBTCTradeExecutor(config)
    
    print("🧪 Testing Enhanced BTC Trade Executor...")
    
    # Test buy order
    buy_trade = executor.place_order("BTCUSD", "buy", 0.001, 43250.50)
    print(f"Buy Trade: {buy_trade.side} {buy_trade.quantity} BTC @ ${buy_trade.price:,.2f}")
    print(f"Status: {buy_trade.status.value}")
    print(f"Fill Price: ${buy_trade.fill_price:,.2f}" if buy_trade.fill_price else "Not filled")
    print(f"Slippage: ${buy_trade.slippage:.2f}")
    
    # Test position info
    position = executor.get_position_info()
    print(f"\nPosition: {position}")
    
    # Test account info
    account = executor.get_account_info()
    print(f"\nAccount: {account}")
    
    # Test trading stats
    stats = executor.get_trading_stats()
    print(f"\nTrading Stats:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Test risk metrics
    risk = executor.get_risk_metrics()
    print(f"\nRisk Metrics: {risk}")
    
    print("✅ Enhanced Trade Executor test completed")