#!/usr/bin/env python3
"""
BTC Scalping Trading Logic - FULLY CORRECTED VERSION
Purpose: Implement €20 to €1M scalping strategy using real-time tick data
FIXES: 1) Lowered momentum and volatility thresholds 2) FIXED position size calculation
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from enum import Enum
from collections import deque


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class ScalpingSignal:
    """Scalping signal for BTC trading"""
    def __init__(self, signal_type: SignalType, confidence: float, reasoning: str, 
                 entry_price: float = 0.0, target_price: float = 0.0, stop_price: float = 0.0):
        self.signal_type = signal_type
        self.confidence = confidence  # 0.0 to 1.0
        self.reasoning = reasoning
        self.entry_price = entry_price
        self.target_price = target_price
        self.stop_price = stop_price
        self.timestamp = datetime.now()


class PositionManager:
    """Manage current trading position for scalping"""
    def __init__(self):
        self.side = None           # 'long', 'short', None
        self.entry_price = None
        self.entry_time = None
        self.quantity = 0.0
        self.target_price = None
        self.stop_price = None
        self.current_balance = 20.0  # Start with €20


class BTCScalpingLogic:
    """
    Core BTC scalping logic for €20 to €1M challenge
    FULLY CORRECTED VERSION - Lower thresholds + Fixed position sizing
    """
    
    def __init__(self, config: Dict = None):
        # €20 to €1M Challenge Configuration
        config = config or {}
        self.profit_target_euros = config.get('profit_target_euros', 8.0)    # €8 target
        self.stop_loss_euros = config.get('stop_loss_euros', 4.0)           # €4 stop
        self.min_confidence = config.get('min_confidence', 0.50)            # Lowered from 0.65
        self.max_position_time = config.get('max_position_time', 20)        # 20 second scalps
        self.risk_per_trade_pct = config.get('risk_per_trade_pct', 2.0)     # 2% risk per trade
        
        # Position management
        self.position = PositionManager()
        
        # Scalping metrics tracking
        self.trades_today = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.min_trade_interval = 3.0  # Minimum 3 seconds between trades
        
        # Technical analysis for scalping
        self.price_buffer = deque(maxlen=30)  # Last 30 ticks for fast analysis
        self.volume_buffer = deque(maxlen=30)
        
        # Performance tracking for €20 to €1M
        self.session_start_balance = 20.0
        self.current_balance = 20.0
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_pnl = 0.0
        
        # Scalping state management
        self.last_signal_time = None
        self.signal_cooldown = 1.0  # 1 second between signals
        
        logging.info(f"✅ BTC Scalping Logic initialized - €20 to €1M Challenge (FULLY CORRECTED)")
        logging.info(f"   Target: €{self.profit_target_euros} | Stop: €{self.stop_loss_euros}")
    
    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed based on scalping rules"""
        
        # Check if already in position
        if self.position.side:
            return False, f"Already in {self.position.side} position"
        
        # Check minimum time between trades (prevent overtrading)
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < self.min_trade_interval:
                return False, f"Too soon since last trade ({time_since_last:.1f}s)"
        
        # Check signal cooldown
        if self.last_signal_time:
            cooldown_time = (datetime.now() - self.last_signal_time).total_seconds()
            if cooldown_time < self.signal_cooldown:
                return False, "Signal cooldown active"
        
        # Check consecutive losses (risk management)
        if self.consecutive_losses >= 3:
            return False, f"Too many consecutive losses: {self.consecutive_losses}"
        
        # Check if we have enough balance to trade
        min_trade_size = 5.0  # Minimum €5 to make meaningful scalp
        if self.current_balance < min_trade_size:
            return False, f"Insufficient balance: €{self.current_balance:.2f}"
        
        return True, "OK"
    
    def evaluate_tick(self, tick_data: Dict, market_metrics: Dict) -> ScalpingSignal:
        """
        Main evaluation method - analyze tick for scalping opportunities
        CORRECTED VERSION with proper position sizing
        """
        
        # Update internal buffers
        self._update_buffers(tick_data)
        
        # Check exit conditions first if in position
        if self.position.side:
            return self._check_exit_conditions(tick_data)
        
        # Check entry conditions if no position
        can_trade, reason = self.can_trade()
        if not can_trade:
            return ScalpingSignal(SignalType.HOLD, 0.0, reason)
        
        # Analyze for scalping entry
        return self._analyze_scalping_entry(tick_data, market_metrics)
    
    def _update_buffers(self, tick_data: Dict):
        """Update price and volume buffers for analysis"""
        self.price_buffer.append(tick_data['price'])
        self.volume_buffer.append(tick_data['size'])
    
    def _calculate_position_size(self, current_price: float) -> float:
        """
        CORRECTED: Calculate appropriate position size for €20 scalping account
        """
        
        # Dynamic position sizing based on account balance
        if self.current_balance <= 30:
            # Very small account - use tiny fixed positions
            return 0.0002  # ~€8.60 at €43,000/BTC
        elif self.current_balance <= 50:
            return 0.0003  # ~€12.90
        elif self.current_balance <= 100:
            return 0.0005  # ~€21.50
        elif self.current_balance <= 200:
            return 0.001   # ~€43.00
        elif self.current_balance <= 500:
            return 0.002   # ~€86.00
        else:
            # Larger accounts - calculate dynamically
            risk_amount = self.current_balance * (self.risk_per_trade_pct / 100)
            max_position_value = risk_amount * 10  # 10:1 ratio for scalping
            position_size = max_position_value / current_price
            return min(position_size, 0.01)  # Cap at 0.01 BTC
    
    def _analyze_scalping_entry(self, tick_data: Dict, market_metrics: Dict) -> ScalpingSignal:
        """Analyze tick data for scalping entry opportunities - CORRECTED position sizing"""
        
        if len(self.price_buffer) < 10:
            return ScalpingSignal(SignalType.HOLD, 0.0, "Insufficient data for analysis")
        
        current_price = tick_data['price']
        
        # Get market metrics for scalping
        momentum_fast = market_metrics.get('momentum_fast', 0)
        momentum_medium = market_metrics.get('momentum_medium', 0)
        volume_spike = market_metrics.get('volume_spike', False)
        price_volatility = market_metrics.get('price_volatility', 0)
        
        # Calculate additional scalping indicators
        indicators = self._calculate_scalping_indicators()
        
        # SCALPING SIGNAL LOGIC - CORRECTED THRESHOLDS
        signal_type = SignalType.HOLD
        confidence = 0.0
        reasoning = ""
        
        # Bullish scalping conditions
        bullish_score = 0
        bearish_score = 0
        
        # CORRECTED: Lower momentum thresholds for more signals
        if momentum_fast > 0.0005:  # LOWERED from 0.002
            bullish_score += 3
        elif momentum_fast > 0.0002:  # LOWERED from 0.001
            bullish_score += 1
        elif momentum_fast < -0.0005:  # LOWERED from -0.002
            bearish_score += 3
        elif momentum_fast < -0.0002:  # LOWERED from -0.001
            bearish_score += 1
        
        # Medium-term momentum confirmation
        if momentum_medium > 0.0003:  # LOWERED from 0.001
            bullish_score += 2
        elif momentum_medium < -0.0003:  # LOWERED from -0.001
            bearish_score += 2
        
        # Volume confirmation (crucial for scalping)
        if volume_spike:
            if momentum_fast > 0:
                bullish_score += 2
            elif momentum_fast < 0:
                bearish_score += 2
        
        # RSI for scalping (faster periods)
        rsi = indicators.get('rsi_fast', 50)
        if 30 < rsi < 45:  # Oversold but not extreme
            bullish_score += 2
        elif 55 < rsi < 70:  # Overbought but not extreme
            bearish_score += 2
        elif rsi < 25:  # Very oversold (reversal opportunity)
            bullish_score += 3
        elif rsi > 75:  # Very overbought (reversal opportunity)
            bearish_score += 3
        
        # Price action patterns
        if indicators.get('bullish_breakout', False):
            bullish_score += 3
        if indicators.get('bearish_breakdown', False):
            bearish_score += 3
        
        # CORRECTED: Much lower volatility requirements
        if price_volatility < 0.0001:  # LOWERED from 0.001
            return ScalpingSignal(SignalType.HOLD, 0.0, "Market too quiet for scalping")
        elif price_volatility > 0.02:  # INCREASED from 0.01 (allow more volatile conditions)
            return ScalpingSignal(SignalType.HOLD, 0.0, "Excessive volatility - too risky")
        
        # CORRECTED: Lower scoring requirements for more signals
        if bullish_score >= 2 and bullish_score > bearish_score:  # LOWERED from 5 and 2
            signal_type = SignalType.BUY
            confidence = min(0.95, 0.4 + (bullish_score / 8))  # ADJUSTED confidence calculation
            reasoning = f"Bullish scalp: Score {bullish_score}, Fast momentum {momentum_fast:.6f}"
            
        elif bearish_score >= 2 and bearish_score > bullish_score:  # LOWERED from 5 and 2
            signal_type = SignalType.SELL
            confidence = min(0.95, 0.4 + (bearish_score / 8))  # ADJUSTED confidence calculation
            reasoning = f"Bearish scalp: Score {bearish_score}, Fast momentum {momentum_fast:.6f}"
        
        else:
            max_score = max(bullish_score, bearish_score)
            confidence = max_score / 8  # ADJUSTED from 10
            reasoning = f"Mixed signals: Bull {bullish_score}, Bear {bearish_score}"
        
        # Apply confidence filter for scalping
        if confidence < self.min_confidence:
            return ScalpingSignal(SignalType.HOLD, confidence, f"Low confidence: {confidence:.2f}")
        
        # CORRECTED: Calculate proper position size and targets
        if signal_type in [SignalType.BUY, SignalType.SELL]:
            entry_price = current_price
            
            # CORRECTED: Use new position size calculation
            position_size = self._calculate_position_size(current_price)
            
            # Calculate targets based on position size
            if signal_type == SignalType.BUY:
                target_price = entry_price + (self.profit_target_euros / position_size)
                stop_price = entry_price - (self.stop_loss_euros / position_size)
            else:  # SELL
                target_price = entry_price - (self.profit_target_euros / position_size)
                stop_price = entry_price + (self.stop_loss_euros / position_size)
            
            self.last_signal_time = datetime.now()
            
            return ScalpingSignal(
                signal_type, confidence, reasoning, 
                entry_price, target_price, stop_price
            )
        
        return ScalpingSignal(SignalType.HOLD, confidence, reasoning)
    
    def _calculate_scalping_indicators(self) -> Dict:
        """Calculate fast indicators for scalping decisions"""
        if len(self.price_buffer) < 10:
            return {}
        
        prices = np.array(list(self.price_buffer))
        volumes = np.array(list(self.volume_buffer))
        
        indicators = {}
        
        # Fast RSI (10 periods for scalping)
        if len(prices) >= 11:
            deltas = np.diff(prices[-11:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
            
            if avg_loss == 0:
                indicators['rsi_fast'] = 100 if avg_gain > 0 else 50
            else:
                rs = avg_gain / avg_loss
                indicators['rsi_fast'] = 100 - (100 / (1 + rs))
        else:
            indicators['rsi_fast'] = 50
        
        # Price breakout detection - CORRECTED with lower thresholds
        if len(prices) >= 10:
            recent_high = np.max(prices[-10:-1])  # Exclude current price
            recent_low = np.min(prices[-10:-1])
            current_price = prices[-1]
            
            indicators['bullish_breakout'] = current_price > recent_high * 1.0005  # LOWERED from 1.001
            indicators['bearish_breakdown'] = current_price < recent_low * 0.9995  # LOWERED from 0.999
        
        # Volume surge detection
        if len(volumes) >= 5:
            recent_volume = volumes[-1]
            avg_volume = np.mean(volumes[-5:-1])
            indicators['volume_surge'] = recent_volume > avg_volume * 1.5  # LOWERED from 2.0
        
        # Micro trend detection (last 5 ticks)
        if len(prices) >= 5:
            micro_trend = np.polyfit(range(5), prices[-5:], 1)[0]  # Slope of last 5 prices
            indicators['micro_trend'] = micro_trend
            indicators['micro_trend_bullish'] = micro_trend > 1.0  # LOWERED from 2.0
            indicators['micro_trend_bearish'] = micro_trend < -1.0  # LOWERED from -2.0
        
        return indicators
    
    def _check_exit_conditions(self, tick_data: Dict) -> ScalpingSignal:
        """Check exit conditions for current scalping position"""
        
        if not self.position.side:
            return ScalpingSignal(SignalType.HOLD, 0.0, "No position to exit")
        
        current_price = tick_data['price']
        current_time = datetime.now()
        
        # Calculate current P&L
        if self.position.side == 'long':
            pnl_ticks = current_price - self.position.entry_price
        else:  # short
            pnl_ticks = self.position.entry_price - current_price
        
        # Convert to euros (approximate)
        position_size = self.position.quantity
        pnl_euros = pnl_ticks * position_size
        
        # Time-based exit (crucial for scalping)
        time_in_position = (current_time - self.position.entry_time).total_seconds()
        if time_in_position > self.max_position_time:
            return ScalpingSignal(
                SignalType.CLOSE, 1.0, 
                f"Time exit: {time_in_position:.0f}s (max {self.max_position_time}s)"
            )
        
        # Profit target hit
        if pnl_euros >= self.profit_target_euros:
            return ScalpingSignal(
                SignalType.CLOSE, 1.0, 
                f"Profit target hit: +€{pnl_euros:.2f}"
            )
        
        # Stop loss hit
        if pnl_euros <= -self.stop_loss_euros:
            return ScalpingSignal(
                SignalType.CLOSE, 1.0, 
                f"Stop loss hit: -€{abs(pnl_euros):.2f}"
            )
        
        # Quick profit protection (take 60% of target if momentum weakens)
        if pnl_euros >= self.profit_target_euros * 0.6:
            if len(self.price_buffer) >= 3:
                recent_momentum = (self.price_buffer[-1] - self.price_buffer[-3]) / self.price_buffer[-3]
                
                if self.position.side == 'long' and recent_momentum < 0.0002:  # LOWERED threshold
                    return ScalpingSignal(
                        SignalType.CLOSE, 0.8, 
                        f"Momentum weakening: +€{pnl_euros:.2f}"
                    )
                elif self.position.side == 'short' and recent_momentum > -0.0002:  # LOWERED threshold
                    return ScalpingSignal(
                        SignalType.CLOSE, 0.8, 
                        f"Momentum weakening: +€{pnl_euros:.2f}"
                    )
        
        # Hold position
        return ScalpingSignal(
            SignalType.HOLD, 0.0, 
            f"Holding: {pnl_euros:+.2f}€ ({time_in_position:.0f}s)"
        )
    
    def update_position(self, action: str, price: float, quantity: float, timestamp: str = None):
        """Update position state for scalping"""
        
        if action in ['buy', 'sell']:
            # Open new position
            self.position.side = 'long' if action == 'buy' else 'short'
            self.position.entry_price = price
            self.position.entry_time = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
            self.position.quantity = quantity
            
            # Calculate targets
            if action == 'buy':
                self.position.target_price = price + (self.profit_target_euros / quantity)
                self.position.stop_price = price - (self.stop_loss_euros / quantity)
            else:
                self.position.target_price = price - (self.profit_target_euros / quantity)
                self.position.stop_price = price + (self.stop_loss_euros / quantity)
            
            self.last_trade_time = datetime.now()
            
            logging.info(f"Scalping position opened: {self.position.side.upper()} @ €{price:.2f}")
            
        elif action == 'close':
            # Close position and update performance
            if self.position.side and self.position.entry_price:
                # Calculate final P&L
                if self.position.side == 'long':
                    pnl_per_unit = price - self.position.entry_price
                else:
                    pnl_per_unit = self.position.entry_price - price
                
                total_pnl = pnl_per_unit * self.position.quantity
                
                # Update balance and tracking
                self.current_balance += total_pnl
                self.daily_pnl += total_pnl
                self.total_trades += 1
                
                if total_pnl > 0:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                
                logging.info(f"Scalping position closed: {total_pnl:+.2f}€ | Balance: €{self.current_balance:.2f}")
            
            # Reset position
            self.position = PositionManager()
            self.position.current_balance = self.current_balance
            self.last_trade_time = datetime.now()
    
    def get_position_info(self) -> Dict:
        """Get current position information for monitoring"""
        if not self.position.side:
            return {
                'has_position': False,
                'balance': self.current_balance,
                'trades_today': self.trades_today,
                'consecutive_losses': self.consecutive_losses
            }
        
        time_in_position = (datetime.now() - self.position.entry_time).total_seconds()
        
        return {
            'has_position': True,
            'side': self.position.side,
            'entry_price': self.position.entry_price,
            'quantity': self.position.quantity,
            'target_price': self.position.target_price,
            'stop_price': self.position.stop_price,
            'time_in_position': time_in_position,
            'entry_time': self.position.entry_time.isoformat(),
            'balance': self.current_balance,
            'trades_today': self.trades_today
        }
    
    def get_scalping_performance(self) -> Dict:
        """Get performance metrics for €20 to €1M challenge"""
        
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        balance_growth = ((self.current_balance - self.session_start_balance) / self.session_start_balance) * 100
        
        # Calculate level in €20 to €1M challenge
        current_level = 0
        level_target = 20.0
        while level_target <= self.current_balance and level_target < 1000000:
            current_level += 1
            level_target *= 2  # Double each level
        
        next_target = level_target if level_target <= 1000000 else 1000000
        progress_to_next = (self.current_balance / next_target) * 100
        
        return {
            'current_balance': self.current_balance,
            'starting_balance': self.session_start_balance,
            'balance_growth_pct': balance_growth,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.total_trades - self.winning_trades,
            'win_rate': win_rate,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            
            # €20 to €1M Challenge specific
            'challenge_level': current_level,
            'next_target': next_target,
            'progress_to_next_pct': progress_to_next,
            'distance_to_million': 1000000 - self.current_balance,
            
            # Risk metrics
            'risk_per_trade_euros': self.current_balance * (self.risk_per_trade_pct / 100),
            'max_daily_loss': self.current_balance * 0.1,  # 10% daily loss limit
        }
    
    def reset_to_twenty_euros(self):
        """Reset balance to €20 for new challenge attempt"""
        
        logging.info(f"🔄 Resetting €20 to €1M challenge")
        logging.info(f"   Previous balance: €{self.current_balance:.2f}")
        logging.info(f"   Trades completed: {self.total_trades}")
        logging.info(f"   Win rate: {(self.winning_trades / max(1, self.total_trades)) * 100:.1f}%")
        
        # Reset all counters
        self.current_balance = 20.0
        self.session_start_balance = 20.0
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trades_today = 0
        
        # Reset position
        self.position = PositionManager()
        self.position.current_balance = 20.0
        
        logging.info("✅ Challenge reset complete - starting fresh with €20")
    
    def should_reset_challenge(self) -> bool:
        """Check if challenge should be reset (balance too low)"""
        return self.current_balance < 5.0  # Reset if below €5
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics for monitoring"""
        
        # Calculate maximum position size based on current balance
        max_risk_per_trade = self.current_balance * (self.risk_per_trade_pct / 100)
        max_position_size = self._calculate_position_size(43000)  # Approximate
        
        # Daily loss limit (10% of current balance)
        daily_loss_limit = self.current_balance * 0.1
        daily_risk_utilization = (abs(self.daily_pnl) / daily_loss_limit) * 100 if self.daily_pnl < 0 else 0
        
        return {
            'current_balance': self.current_balance,
            'risk_per_trade_euros': max_risk_per_trade,
            'max_position_size': max_position_size,
            'daily_loss_limit': daily_loss_limit,
            'daily_pnl': self.daily_pnl,
            'daily_risk_utilization_pct': daily_risk_utilization,
            'consecutive_losses': self.consecutive_losses,
            'should_reset': self.should_reset_challenge(),
            'trades_until_rest': max(0, 10 - (self.trades_today % 10)),  # Rest every 10 trades
        }


if __name__ == "__main__":
    # Test BTC scalping logic
    config = {
        'profit_target_euros': 8.0,
        'stop_loss_euros': 4.0,
        'min_confidence': 0.50,  # CORRECTED
        'max_position_time': 20
    }
    
    logic = BTCScalpingLogic(config)
    
    # Sample tick data
    tick_data = {
        'price': 43250.50,
        'size': 1.5,
        'timestamp': datetime.now()
    }
    
    # Sample market metrics with lower values (realistic)
    market_metrics = {
        'momentum_fast': 0.0008,  # Realistic low momentum
        'momentum_medium': 0.0005,
        'volume_spike': True,
        'price_volatility': 0.0003  # Low but acceptable volatility
    }
    
    print("🧪 Testing FULLY CORRECTED BTC Scalping Logic...")
    
    # Test signal generation
    signal = logic.evaluate_tick(tick_data, market_metrics)
    print(f"Signal: {signal.signal_type.value} | Confidence: {signal.confidence:.2f}")
    print(f"Reasoning: {signal.reasoning}")
    
    # Test position size calculation
    position_size = logic._calculate_position_size(43250.50)
    print(f"Position size for €20 account: {position_size:.6f} BTC (€{position_size * 43250.50:.2f})")
    
    # Test performance metrics
    performance = logic.get_scalping_performance()
    print(f"\n📊 Performance: {performance}")
    
    print("✅ FULLY CORRECTED BTC Scalping Logic test completed")