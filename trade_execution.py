#!/usr/bin/env python3
"""
BTC Trade Execution - Core File 3/4
Purpose: Handle Alpaca API interaction for fast BTC order execution
Ensures quick, efficient execution of scalping signals
"""

import logging
import time
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BTCOrder:
    """BTC order structure for scalping execution"""
    symbol: str
    side: str          # 'buy' or 'sell'
    quantity: float    # BTC amount
    price: float       # Entry price
    order_type: str = "market"
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = ""
    timestamp: str = ""
    fill_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"BTC_{self.side}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class BTCTradeExecutor:
    """
    Fast BTC trade executor for scalping
    Handles order placement and execution via Alpaca API
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # API Configuration
        self.paper_trading = config.get('paper_trading', True)
        self.api_key = config.get('api_key', "")
        self.secret_key = config.get('secret_key', "")
        self.alpaca_api = None
        
        # Execution settings for scalping
        self.max_slippage_euros = 2.0      # Max €2 slippage per trade
        self.order_timeout = 5.0           # 5 second timeout for scalping
        self.max_retries = 2               # Quick retries for scalping
        
        # Simulated account for demo/testing
        self.simulated_balance = 20.0      # Start with €20
        self.simulated_btc_holdings = 0.0
        
        # Performance tracking
        self.total_orders = 0
        self.successful_orders = 0
        self.total_slippage = 0.0
        self.total_commission = 0.0
        self.execution_times = []
        
        # Order management
        self.pending_orders = {}
        self.last_order_time = None
        self.min_order_interval = 1.0     # 1 second minimum between orders
        
        # Initialize API connection
        if self.api_key and self.secret_key and self.api_key != 'YOUR_ALPACA_API_KEY':
            self._initialize_alpaca_api()
        else:
            logging.info("🎮 BTC Executor in simulation mode")
            print("📄 Trade executor: Simulation mode (demo trading)")
    
    def _initialize_alpaca_api(self):
        """Initialize Alpaca API for BTC trading"""
        try:
            import alpaca_trade_api as tradeapi
            
            base_url = 'https://paper-api.alpaca.markets' if self.paper_trading else 'https://api.alpaca.markets'
            
            self.alpaca_api = tradeapi.REST(
                self.api_key,
                self.secret_key,
                base_url,
                api_version='v2'
            )
            
            # Test connection
            account = self.alpaca_api.get_account()
            
            mode = "📄 PAPER" if self.paper_trading else "🔴 LIVE"
            logging.info(f"✅ Alpaca API connected - {mode} trading")
            print(f"✅ Alpaca connected - {mode} mode")
            print(f"💰 Account value: ${float(account.portfolio_value):,.2f}")
            
        except ImportError:
            logging.warning("alpaca-trade-api not installed - using simulation")
            print("⚠️ alpaca-trade-api not installed - using simulation")
            self.alpaca_api = None
        except Exception as e:
            logging.error(f"Alpaca connection failed: {e}")
            print(f"❌ Alpaca connection failed: {e}")
            self.alpaca_api = None
    
    def place_order(self, symbol: str, side: str, quantity: float, 
                   current_price: float, order_type: str = "market") -> BTCOrder:
        """
        Place BTC order with fast execution for scalping
        Returns order object with execution details
        """
        
        # Create order object
        order = BTCOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=current_price,
            order_type=order_type
        )
        
        self.total_orders += 1
        
        # Pre-execution validation
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            return order
        
        # Execute order
        start_time = time.time()
        
        try:
            if self.alpaca_api:
                executed_order = self._execute_alpaca_order(order)
            else:
                executed_order = self._execute_simulated_order(order)
            
            # Track execution time
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            
            # Update last order time
            self.last_order_time = time.time()
            
            return executed_order
            
        except Exception as e:
            logging.error(f"Order execution failed: {e}")
            order.status = OrderStatus.FAILED
            return order
    
    def _validate_order(self, order: BTCOrder) -> bool:
        """Validate order before execution"""
        
        # Check minimum order size
        if order.quantity < 0.0001:  # Minimum BTC amount
            logging.warning(f"Order too small: {order.quantity} BTC")
            return False
        
        # Check maximum order size (risk management)
        max_order_size = 0.01  # Max 0.01 BTC per order
        if order.quantity > max_order_size:
            logging.warning(f"Order too large: {order.quantity} > {max_order_size}")
            return False
        
        # Check order frequency (prevent overtrading)
        if self.last_order_time:
            time_since_last = time.time() - self.last_order_time
            if time_since_last < self.min_order_interval:
                logging.warning(f"Order too frequent: {time_since_last:.1f}s")
                return False
        
        # Check simulated balance
        if not self.alpaca_api:
            order_value = order.quantity * order.price
            if order.side == 'buy':
                if self.simulated_balance < order_value:
                    logging.warning(f"Insufficient balance: €{order_value:.2f} needed")
                    return False
            else:  # sell
                if self.simulated_btc_holdings < order.quantity:
                    logging.warning(f"Insufficient BTC: {order.quantity} needed")
                    return False
        
        return True
    
    def _execute_alpaca_order(self, order: BTCOrder) -> BTCOrder:
        """Execute order via Alpaca API"""
        
        for attempt in range(self.max_retries):
            try:
                # Submit market order for fast execution
                alpaca_order = self.alpaca_api.submit_order(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=order.side,
                    type='market',
                    time_in_force='day'
                )
                
                # Wait for fill with timeout
                start_wait = time.time()
                while time.time() - start_wait < self.order_timeout:
                    updated_order = self.alpaca_api.get_order(alpaca_order.id)
                    
                    if updated_order.status == 'filled':
                        # Order successfully filled
                        fill_price = float(updated_order.filled_avg_price)
                        order.fill_price = fill_price
                        order.status = OrderStatus.FILLED
                        order.slippage = abs(fill_price - order.price)
                        order.commission = float(updated_order.commission) if updated_order.commission else 0.0
                        
                        self.successful_orders += 1
                        self.total_slippage += order.slippage
                        self.total_commission += order.commission
                        
                        logging.info(f"✅ Alpaca order filled: {order.side.upper()} {order.quantity} BTC @ ${fill_price:,.2f}")
                        return order
                    
                    elif updated_order.status in ['rejected', 'canceled']:
                        order.status = OrderStatus.REJECTED
                        logging.warning(f"❌ Alpaca order rejected: {alpaca_order.id}")
                        return order
                    
                    time.sleep(0.1)  # Check every 100ms
                
                # Timeout - order might still be pending
                order.status = OrderStatus.PENDING
                logging.warning(f"⚠️ Order timeout: {alpaca_order.id}")
                return order
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if 'insufficient' in error_msg:
                    order.status = OrderStatus.REJECTED
                    logging.warning(f"❌ Insufficient funds: {e}")
                    return order
                elif attempt < self.max_retries - 1:
                    logging.warning(f"⚠️ Order attempt {attempt + 1} failed: {e}")
                    time.sleep(0.5)  # Brief wait before retry
                    continue
                else:
                    order.status = OrderStatus.FAILED
                    logging.error(f"❌ Order failed after {self.max_retries} attempts: {e}")
                    return order
        
        return order
    
    def _execute_simulated_order(self, order: BTCOrder) -> BTCOrder:
        """Execute simulated order for demo trading"""
        
        try:
            # Simulate realistic execution delay
            execution_delay = np.random.uniform(0.05, 0.2)  # 50-200ms
            time.sleep(execution_delay)
            
            # Calculate realistic slippage for BTC
            base_slippage = np.random.uniform(0.5, 2.0)  # €0.50 - €2.00
            slippage_direction = 1 if order.side == 'buy' else -1
            
            # Higher slippage during volatile periods (simulated)
            volatility_factor = 1 + np.random.uniform(0, 0.3)
            total_slippage = base_slippage * volatility_factor * slippage_direction
            
            # Apply slippage to fill price
            fill_price = max(1.0, order.price + total_slippage)
            
            # Simulate occasional rejection (1% chance)
            if np.random.random() < 0.01:
                order.status = OrderStatus.REJECTED
                logging.info(f"🎮 Simulated order rejection")
                return order
            
            # Order filled successfully
            order.fill_price = fill_price
            order.status = OrderStatus.FILLED
            order.slippage = abs(total_slippage)
            order.commission = order.quantity * fill_price * 0.001  # 0.1% commission
            
            # Update simulated account
            self._update_simulated_account(order)
            
            # Track performance
            self.successful_orders += 1
            self.total_slippage += order.slippage
            self.total_commission += order.commission
            
            # Log execution
            slippage_str = f"(${total_slippage:+.2f})" if abs(total_slippage) > 0.1 else ""
            logging.info(f"✅ Simulated order: {order.side.upper()} {order.quantity} BTC @ ${fill_price:,.2f} {slippage_str}")
            
            return order
            
        except Exception as e:
            logging.error(f"Simulated execution error: {e}")
            order.status = OrderStatus.FAILED
            return order
    
    def _update_simulated_account(self, order: BTCOrder):
        """Update simulated account balances"""
        
        if order.status != OrderStatus.FILLED:
            return
        
        trade_value = order.quantity * order.fill_price
        
        if order.side == 'buy':
            self.simulated_balance -= (trade_value + order.commission)
            self.simulated_btc_holdings += order.quantity
        else:  # sell
            self.simulated_balance += (trade_value - order.commission)
            self.simulated_btc_holdings -= order.quantity
        
        # Ensure no negative balances
        self.simulated_balance = max(0, self.simulated_balance)
        self.simulated_btc_holdings = max(0, self.simulated_btc_holdings)
    
    def close_position(self, symbol: str, quantity: float, current_price: float) -> Optional[BTCOrder]:
        """Close BTC position with market order"""
        
        if quantity <= 0:
            return None
        
        # Determine close side based on position
        if self.simulated_btc_holdings > 0:
            close_side = 'sell'
            close_quantity = min(quantity, self.simulated_btc_holdings)
        else:
            # Short position (not typical for demo, but handled)
            close_side = 'buy'
            close_quantity = quantity
        
        # Place closing order
        close_order = self.place_order(symbol, close_side, close_quantity, current_price)
        
        if close_order.status == OrderStatus.FILLED:
            logging.info(f"Position closed: {close_side.upper()} {close_quantity} BTC @ ${close_order.fill_price:,.2f}")
        
        return close_order
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        
        if self.alpaca_api:
            try:
                account = self.alpaca_api.get_account()
                positions = self.alpaca_api.list_positions()
                
                # Get BTC position if exists
                btc_position = None
                for pos in positions:
                    if pos.symbol == 'BTCUSD':
                        btc_position = pos
                        break
                
                return {
                    'balance': float(account.portfolio_value),
                    'cash': float(account.cash),
                    'buying_power': float(account.buying_power),
                    'btc_holdings': float(btc_position.qty) if btc_position else 0.0,
                    'btc_market_value': float(btc_position.market_value) if btc_position else 0.0,
                    'account_status': account.status,
                    'connection_type': 'alpaca_live' if not self.paper_trading else 'alpaca_paper'
                }
            except Exception as e:
                logging.warning(f"Failed to get live account info: {e}")
        
        # Return simulated account info
        btc_value = self.simulated_btc_holdings * 43000  # Approximate BTC value
        total_value = self.simulated_balance + btc_value
        
        return {
            'balance': total_value,
            'cash': self.simulated_balance,
            'buying_power': self.simulated_balance,
            'btc_holdings': self.simulated_btc_holdings,
            'btc_market_value': btc_value,
            'account_status': 'simulated',
            'connection_type': 'simulation'
        }
    
    def get_execution_stats(self) -> Dict:
        """Get execution performance statistics"""
        
        success_rate = (self.successful_orders / max(1, self.total_orders)) * 100
        avg_slippage = self.total_slippage / max(1, self.successful_orders)
        avg_commission = self.total_commission / max(1, self.successful_orders)
        avg_execution_time = np.mean(self.execution_times) if self.execution_times else 0
        
        return {
            'total_orders': self.total_orders,
            'successful_orders': self.successful_orders,
            'success_rate': success_rate,
            'avg_slippage': avg_slippage,
            'total_slippage': self.total_slippage,
            'avg_commission': avg_commission,
            'total_commission': self.total_commission,
            'avg_execution_time': avg_execution_time,
            'connection_type': 'alpaca' if self.alpaca_api else 'simulation'
        }
    
    def get_position_info(self) -> Dict:
        """Get current position information"""
        
        if self.alpaca_api:
            try:
                positions = self.alpaca_api.list_positions()
                for pos in positions:
                    if pos.symbol == 'BTCUSD':
                        return {
                            'has_position': True,
                            'quantity': float(pos.qty),
                            'side': 'long' if float(pos.qty) > 0 else 'short',
                            'avg_price': float(pos.avg_cost),
                            'market_value': float(pos.market_value),
                            'unrealized_pnl': float(pos.unrealized_pnl),
                            'symbol': pos.symbol
                        }
            except Exception as e:
                logging.warning(f"Failed to get position info: {e}")
        
        # Return simulated position info
        has_position = self.simulated_btc_holdings > 0.0001
        
        return {
            'has_position': has_position,
            'quantity': self.simulated_btc_holdings,
            'side': 'long' if has_position else 'none',
            'avg_price': 43000.0,  # Approximate
            'market_value': self.simulated_btc_holdings * 43000,
            'unrealized_pnl': 0.0,  # Simplified for demo
            'symbol': 'BTCUSD'
        }
    
    def is_market_open(self) -> bool:
        """Check if BTC market is open (always true for crypto)"""
        return True  # Crypto markets are 24/7
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        
        if self.alpaca_api:
            try:
                self.alpaca_api.cancel_order(order_id)
                logging.info(f"Order cancelled: {order_id}")
                return True
            except Exception as e:
                logging.error(f"Failed to cancel order {order_id}: {e}")
                return False
        
        # Simulated cancellation
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            return True
        
        return False


if __name__ == "__main__":
    # Test BTC trade executor
    config = {
        'paper_trading': True,
        'api_key': 'YOUR_ALPACA_API_KEY',
        'secret_key': 'YOUR_ALPACA_SECRET_KEY'
    }
    
    executor = BTCTradeExecutor(config)
    
    print("🧪 Testing BTC Trade Executor...")
    
    # Test buy order
    buy_order = executor.place_order("BTCUSD", "buy", 0.001, 43250.50)
    print(f"\nBuy Order Results:")
    print(f"   Status: {buy_order.status.value}")
    print(f"   Fill Price: ${buy_order.fill_price:,.2f}" if buy_order.fill_price else "   Not filled")
    print(f"   Slippage: ${buy_order.slippage:.2f}")
    print(f"   Commission: ${buy_order.commission:.2f}")
    
    # Test account info
    account = executor.get_account_info()
    print(f"\nAccount Information:")
    print(f"   Balance: €{account['balance']:,.2f}")
    print(f"   Cash: €{account['cash']:,.2f}")
    print(f"   BTC Holdings: {account['btc_holdings']:.6f}")
    print(f"   Connection: {account['connection_type']}")
    
    # Test execution statistics
    stats = executor.get_execution_stats()
    print(f"\nExecution Statistics:")
    print(f"   Total Orders: {stats['total_orders']}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")
    print(f"   Avg Slippage: ${stats['avg_slippage']:.2f}")
    print(f"   Avg Execution Time: {stats['avg_execution_time']:.3f}s")
    
    # Test position info
    position = executor.get_position_info()
    print(f"\nPosition Information:")
    print(f"   Has Position: {position['has_position']}")
    if position['has_position']:
        print(f"   Side: {position['side']}")
        print(f"   Quantity: {position['quantity']:.6f} BTC")
        print(f"   Market Value: €{position['market_value']:,.2f}")
    
    print("\n✅ BTC Trade Executor test completed")