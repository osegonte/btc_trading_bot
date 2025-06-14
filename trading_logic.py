#!/usr/bin/env python3
"""
Enhanced Trading Logic for BTC/USD with Position Management Integration
Includes: Advanced indicators, ML integration, risk management
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from enum import Enum


class SignalStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class TradingSignal:
    """Enhanced trading signal with detailed information"""
    def __init__(self, signal_type: str, confidence: float, reasoning: str, 
                 strength: SignalStrength = SignalStrength.MODERATE, indicators: Dict = None):
        self.signal_type = signal_type  # 'buy', 'sell', 'hold', 'close'
        self.confidence = confidence    # 0.0 to 1.0
        self.reasoning = reasoning
        self.strength = strength
        self.indicators = indicators or {}
        self.timestamp = datetime.now()


class PositionState:
    """Position state management"""
    def __init__(self):
        self.side = None           # 'long', 'short', None
        self.entry_price = None
        self.entry_time = None
        self.quantity = 0.0
        self.unrealized_pnl = 0.0
        self.max_profit = 0.0
        self.max_loss = 0.0


class EnhancedBTCTradingLogic:
    """Enhanced trading logic for BTC/USD with integrated position management"""
    
    def __init__(self, config: Dict = None):
        # Configuration
        config = config or {}
        self.profit_target_ticks = config.get('profit_target_ticks', 10)
        self.stop_loss_ticks = config.get('stop_loss_ticks', 5)
        self.tick_size = config.get('tick_size', 1.0)
        self.min_confidence = config.get('min_confidence', 0.75)
        self.max_position_time = config.get('max_position_time', 30)
        
        # Position management
        self.position = PositionState()
        self.trades_today = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.min_trade_interval = config.get('min_trade_interval', 3.0)
        
        # Enhanced technical analysis
        self.price_history = []
        self.volume_history = []
        self.indicator_history = []
        self.rsi_period = 14
        self.bollinger_period = 20
        self.ema_periods = [5, 10, 20]
        
        # Performance tracking
        self.total_signals = 0
        self.successful_signals = 0
        self.signal_performance = {'buy': [], 'sell': []}
        
        # Risk management
        self.daily_pnl = 0.0
        self.max_daily_loss = config.get('max_daily_loss', 500.0)
        self.position_size_multiplier = 1.0
        
        logging.info(f"✅ Enhanced BTC trading logic initialized")
        logging.info(f"   Target: {self.profit_target_ticks} ticks | Stop: {self.stop_loss_ticks} ticks")
    
    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed based on various conditions"""
        
        # Check if we already have a position
        if self.position.side:
            return False, f"Already have {self.position.side} position"
        
        # Check minimum time between trades
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < self.min_trade_interval:
                return False, f"Too soon since last trade ({time_since_last:.1f}s)"
        
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            return False, f"Daily loss limit reached: ${self.daily_pnl:.2f}"
        
        # Check consecutive losses
        if self.consecutive_losses >= 5:
            return False, f"Too many consecutive losses: {self.consecutive_losses}"
        
        return True, "OK"
    
    def evaluate_tick(self, tick_data: Dict, market_analysis: Dict, ml_signal=None) -> TradingSignal:
        """Main evaluation method with enhanced logic"""
        
        # Update internal state
        self._update_internal_state(tick_data, market_analysis)
        
        # Check exit conditions first if in position
        if self.position.side:
            exit_signal = self._check_exit_conditions(tick_data, market_analysis)
            if exit_signal.signal_type == 'close':
                return exit_signal
        
        # Check entry conditions if no position
        if not self.position.side:
            can_trade, reason = self.can_trade()
            if not can_trade:
                return TradingSignal('hold', 0.0, reason)
            
            return self._check_entry_conditions(tick_data, market_analysis, ml_signal)
        
        return TradingSignal('hold', 0.0, 'No action needed')
    
    def _update_internal_state(self, tick_data: Dict, market_analysis: Dict):
        """Update internal state with new data"""
        current_price = tick_data.get('price', 0)
        current_volume = tick_data.get('size', 0)
        
        # Update price and volume history
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)
        
        # Keep only recent history
        if len(self.price_history) > 100:
            self.price_history.pop(0)
        if len(self.volume_history) > 100:
            self.volume_history.pop(0)
        
        # Update position P&L if in position
        if self.position.side:
            self._update_position_pnl(current_price)
    
    def _update_position_pnl(self, current_price: float):
        """Update position P&L and tracking"""
        if not self.position.side or not self.position.entry_price:
            return
        
        if self.position.side == 'long':
            pnl = (current_price - self.position.entry_price) * self.position.quantity
        else:  # short
            pnl = (self.position.entry_price - current_price) * self.position.quantity
        
        self.position.unrealized_pnl = pnl
        
        # Track max profit/loss for trailing stops
        if pnl > self.position.max_profit:
            self.position.max_profit = pnl
        if pnl < self.position.max_loss:
            self.position.max_loss = pnl
    
    def _check_entry_conditions(self, tick_data: Dict, market_analysis: Dict, ml_signal=None) -> TradingSignal:
        """Enhanced entry condition checking"""
        
        current_price = tick_data.get('price', 0)
        spread = tick_data.get('spread', 0)
        
        # Basic filters
        if spread > 2.0:  # Wider spread tolerance for BTC
            return TradingSignal('hold', 0.0, 'Spread too wide for BTC')
        
        if market_analysis.get('price_volatility', 0) > 0.5:
            return TradingSignal('hold', 0.0, 'Extreme volatility')
        
        # Calculate technical indicators
        indicators = self._calculate_technical_indicators()
        
        # Enhanced scoring system
        bullish_score, bearish_score = self._calculate_signal_scores(market_analysis, indicators)
        
        # ML integration
        ml_boost = 0
        if ml_signal and ml_signal.confidence > 0.75:
            if ml_signal.signal == 'buy':
                bullish_score += 3
                ml_boost = ml_signal.confidence
            elif ml_signal.signal == 'sell':
                bearish_score += 3
                ml_boost = ml_signal.confidence
        
        # Determine signal
        signal_type = 'hold'
        confidence = 0.0
        reasoning = ""
        strength = SignalStrength.WEAK
        
        max_score = 20  # Maximum possible score
        
        if bullish_score >= 12 and bullish_score > bearish_score + 3:
            confidence = min(0.95, 0.6 + (bullish_score / max_score) + (ml_boost * 0.2))
            signal_type = 'buy'
            strength = self._get_signal_strength(bullish_score, max_score)
            reasoning = f"Bullish: {bullish_score}/{max_score} | RSI: {indicators.get('rsi', 0):.1f}"
            
        elif bearish_score >= 12 and bearish_score > bullish_score + 3:
            confidence = min(0.95, 0.6 + (bearish_score / max_score) + (ml_boost * 0.2))
            signal_type = 'sell'
            strength = self._get_signal_strength(bearish_score, max_score)
            reasoning = f"Bearish: {bearish_score}/{max_score} | RSI: {indicators.get('rsi', 0):.1f}"
        else:
            max_combined = max(bullish_score, bearish_score)
            confidence = max_combined / max_score
            reasoning = f"Mixed signals: Bull {bullish_score}, Bear {bearish_score}"
        
        # Apply confidence filter
        if confidence < self.min_confidence:
            return TradingSignal('hold', confidence, f"Confidence too low: {confidence:.2f}")
        
        return TradingSignal(signal_type, confidence, reasoning, strength, indicators)
    
    def _calculate_signal_scores(self, market_analysis: Dict, indicators: Dict) -> tuple[int, int]:
        """Calculate bullish and bearish scores based on multiple factors"""
        
        bullish_score = 0
        bearish_score = 0
        
        # Price momentum analysis
        momentum = market_analysis.get('momentum', 0)
        momentum_long = market_analysis.get('momentum_long', 0)
        
        if momentum > 0.003:  # Strong upward momentum
            bullish_score += 4
        elif momentum > 0.001:  # Moderate upward momentum
            bullish_score += 2
        elif momentum < -0.003:  # Strong downward momentum
            bearish_score += 4
        elif momentum < -0.001:  # Moderate downward momentum
            bearish_score += 2
        
        # Long-term momentum
        if momentum_long > 0.005:
            bullish_score += 2
        elif momentum_long < -0.005:
            bearish_score += 2
        
        # Price changes
        price_change_5 = market_analysis.get('price_change_5', 0)
        price_change_10 = market_analysis.get('price_change_10', 0)
        
        if price_change_5 > 0.15:  # 0.15% in 5 ticks
            bullish_score += 3
        elif price_change_5 > 0.05:
            bullish_score += 1
        elif price_change_5 < -0.15:
            bearish_score += 3
        elif price_change_5 < -0.05:
            bearish_score += 1
        
        # RSI analysis
        rsi = indicators.get('rsi', 50)
        if 30 < rsi < 70:  # Good trading range
            if rsi > 50:
                bullish_score += 1
            else:
                bearish_score += 1
        elif rsi < 30:  # Oversold - potential reversal
            bullish_score += 3
        elif rsi > 70:  # Overbought - potential reversal
            bearish_score += 3
        
        # Bollinger Bands
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position < 0.2:  # Near lower band
            bullish_score += 2
        elif bb_position > 0.8:  # Near upper band
            bearish_score += 2
        elif 0.3 < bb_position < 0.7:  # Middle range
            bullish_score += 1
            bearish_score += 1
        
        # Moving averages
        if market_analysis.get('price_above_sma5', False):
            bullish_score += 1
        else:
            bearish_score += 1
            
        if market_analysis.get('sma5_above_sma10', False):
            bullish_score += 2
        else:
            bearish_score += 2
        
        # Volume analysis
        volume_trend = market_analysis.get('volume_trend', 0)
        volume_ratio = market_analysis.get('volume_ratio', 1)
        
        if volume_trend > 0.2 and volume_ratio > 1.5:  # Increasing volume
            if momentum > 0:
                bullish_score += 2
            else:
                bearish_score += 2
        
        # EMA analysis
        ema_signals = indicators.get('ema_signals', {})
        if ema_signals.get('bullish_crossover', False):
            bullish_score += 3
        if ema_signals.get('bearish_crossover', False):
            bearish_score += 3
        
        return bullish_score, bearish_score
    
    def _get_signal_strength(self, score: int, max_score: int) -> SignalStrength:
        """Determine signal strength based on score"""
        ratio = score / max_score
        if ratio >= 0.85:
            return SignalStrength.VERY_STRONG
        elif ratio >= 0.70:
            return SignalStrength.STRONG
        elif ratio >= 0.55:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _check_exit_conditions(self, tick_data: Dict, market_analysis: Dict) -> TradingSignal:
        """Enhanced exit condition checking"""
        
        if not self.position.side or not self.position.entry_price:
            return TradingSignal('hold', 0.0, 'No position')
        
        current_price = tick_data.get('price', 0)
        current_time = datetime.now()
        
        # Calculate P&L in ticks
        if self.position.side == 'long':
            pnl_ticks = (current_price - self.position.entry_price) / self.tick_size
        else:  # short
            pnl_ticks = (self.position.entry_price - current_price) / self.tick_size
        
        # Time-based exit
        if self.position.entry_time:
            time_in_position = (current_time - self.position.entry_time).total_seconds()
            if time_in_position > self.max_position_time:
                return TradingSignal('close', 1.0, f'Time exit: {time_in_position:.0f}s')
        
        # Profit target
        if pnl_ticks >= self.profit_target_ticks:
            return TradingSignal('close', 1.0, f'Profit target: +{pnl_ticks:.1f} ticks')
        
        # Stop loss
        if pnl_ticks <= -self.stop_loss_ticks:
            return TradingSignal('close', 1.0, f'Stop loss: {pnl_ticks:.1f} ticks')
        
        # Dynamic exits based on indicators
        indicators = self._calculate_technical_indicators()
        
        # RSI extremes
        rsi = indicators.get('rsi', 50)
        if self.position.side == 'long' and rsi > 80:
            return TradingSignal('close', 0.8, f'RSI overbought exit: {rsi:.1f}')
        elif self.position.side == 'short' and rsi < 20:
            return TradingSignal('close', 0.8, f'RSI oversold exit: {rsi:.1f}')
        
        # Momentum reversal
        momentum = market_analysis.get('momentum', 0)
        if self.position.side == 'long' and momentum < -0.004:
            return TradingSignal('close', 0.7, f'Momentum reversal: {momentum:.4f}')
        elif self.position.side == 'short' and momentum > 0.004:
            return TradingSignal('close', 0.7, f'Momentum reversal: {momentum:.4f}')
        
        # Trailing stop based on max profit
        if self.position.max_profit > self.profit_target_ticks * self.tick_size * 0.5:
            current_from_peak = self.position.max_profit - self.position.unrealized_pnl
            trailing_threshold = self.position.max_profit * 0.3  # 30% from peak
            
            if current_from_peak > trailing_threshold:
                return TradingSignal('close', 0.9, f'Trailing stop: {current_from_peak:.2f} from peak')
        
        return TradingSignal('hold', 0.0, f'Hold: {pnl_ticks:+.1f} ticks')
    
    def _calculate_technical_indicators(self) -> Dict:
        """Calculate comprehensive technical indicators"""
        if len(self.price_history) < 20:
            return {}
        
        prices = np.array(self.price_history)
        indicators = {}
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(prices)
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(prices)
        indicators.update(bb_data)
        
        # EMAs
        ema_data = self._calculate_emas(prices)
        indicators.update(ema_data)
        
        # MACD
        macd_data = self._calculate_macd(prices)
        indicators.update(macd_data)
        
        # Support/Resistance
        sr_data = self._calculate_support_resistance(prices)
        indicators.update(sr_data)
        
        return indicators
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> Dict:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            current_price = prices[-1]
            return {
                'bb_upper': current_price * 1.02,
                'bb_middle': current_price,
                'bb_lower': current_price * 0.98,
                'bb_position': 0.5,
                'bb_width': 0.04
            }
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        
        current_price = prices[-1]
        bb_position = (current_price - lower) / (upper - lower) if upper != lower else 0.5
        bb_width = (upper - lower) / sma if sma > 0 else 0
        
        return {
            'bb_upper': upper,
            'bb_middle': sma,
            'bb_lower': lower,
            'bb_position': bb_position,
            'bb_width': bb_width
        }
    
    def _calculate_emas(self, prices: np.ndarray) -> Dict:
        """Calculate EMAs and crossover signals"""
        ema_data = {}
        
        for period in self.ema_periods:
            if len(prices) >= period:
                alpha = 2 / (period + 1)
                ema = prices[-period]  # Start with SMA
                for price in prices[-period+1:]:
                    ema = alpha * price + (1 - alpha) * ema
                ema_data[f'ema_{period}'] = ema
        
        # Crossover signals
        if 'ema_5' in ema_data and 'ema_10' in ema_data:
            # Check if we have previous values for crossover detection
            if len(self.indicator_history) > 0:
                prev_indicators = self.indicator_history[-1]
                prev_ema5 = prev_indicators.get('ema_5', ema_data['ema_5'])
                prev_ema10 = prev_indicators.get('ema_10', ema_data['ema_10'])
                
                # Detect crossovers
                bullish_cross = (prev_ema5 <= prev_ema10 and ema_data['ema_5'] > ema_data['ema_10'])
                bearish_cross = (prev_ema5 >= prev_ema10 and ema_data['ema_5'] < ema_data['ema_10'])
                
                ema_data['ema_signals'] = {
                    'bullish_crossover': bullish_cross,
                    'bearish_crossover': bearish_cross
                }
        
        return ema_data
    
    def _calculate_macd(self, prices: np.ndarray) -> Dict:
        """Calculate MACD"""
        if len(prices) < 26:
            return {}
        
        # Calculate EMAs for MACD
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        
        macd_line = ema_12 - ema_26
        signal_line = self._ema(np.array([macd_line]), 9)  # Simplified
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }
    
    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate single EMA value"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2 / (period + 1)
        ema = prices[-period]
        for price in prices[-period+1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def _calculate_support_resistance(self, prices: np.ndarray) -> Dict:
        """Calculate support and resistance levels"""
        if len(prices) < 20:
            return {}
        
        recent_high = np.max(prices[-20:])
        recent_low = np.min(prices[-20:])
        current_price = prices[-1]
        
        # Distance to support/resistance
        support_distance = (current_price - recent_low) / recent_low if recent_low > 0 else 1
        resistance_distance = (recent_high - current_price) / recent_high if recent_high > 0 else 1
        
        return {
            'resistance_level': recent_high,
            'support_level': recent_low,
            'near_support': support_distance < 0.005,
            'near_resistance': resistance_distance < 0.005,
            'support_strength': 1 / (support_distance + 0.001),
            'resistance_strength': 1 / (resistance_distance + 0.001)
        }
    
    def update_position(self, action: str, price: float, quantity: float = 0.001, timestamp: str = None):
        """Update position state"""
        
        if action in ['buy', 'sell']:
            self.position.side = 'long' if action == 'buy' else 'short'
            self.position.entry_price = price
            self.position.entry_time = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
            self.position.quantity = quantity
            self.position.unrealized_pnl = 0.0
            self.position.max_profit = 0.0
            self.position.max_loss = 0.0
            
            self.last_trade_time = datetime.now()
            
            logging.info(f"Position opened: {self.position.side.upper()} {quantity} BTC @ ${price:,.2f}")
            
        elif action == 'close':
            if self.position.side and self.position.entry_price:
                # Calculate final P&L
                if self.position.side == 'long':
                    pnl_ticks = (price - self.position.entry_price) / self.tick_size
                else:
                    pnl_ticks = (self.position.entry_price - price) / self.tick_size
                
                pnl_usd = pnl_ticks * self.tick_size
                
                # Update performance tracking
                self.total_signals += 1
                if pnl_usd > 0:
                    self.successful_signals += 1
                    self.consecutive_losses = 0
                    self.signal_performance[self.position.side].append(pnl_usd)
                else:
                    self.consecutive_losses += 1
                
                self.daily_pnl += pnl_usd
                
                # Adjust position size based on performance
                self._adjust_position_size_multiplier(pnl_usd)
                
                logging.info(f"Position closed: {pnl_ticks:+.1f} ticks (${pnl_usd:+.2f})")
            
            # Reset position
            self.position = PositionState()
            self.last_trade_time = datetime.now()
    
    def _adjust_position_size_multiplier(self, pnl: float):
        """Adjust position size based on recent performance"""
        if pnl > 0:
            self.position_size_multiplier = min(1.5, self.position_size_multiplier * 1.05)
        else:
            self.position_size_multiplier = max(0.5, self.position_size_multiplier * 0.95)
    
    def get_position_info(self) -> Dict:
        """Get comprehensive position information"""
        return {
            'has_position': bool(self.position.side),
            'side': self.position.side or 'none',
            'entry_price': self.position.entry_price or 0.0,
            'quantity': self.position.quantity,
            'unrealized_pnl': self.position.unrealized_pnl,
            'max_profit': self.position.max_profit,
            'max_loss': self.position.max_loss,
            'time_in_position': (datetime.now() - self.position.entry_time).total_seconds() if self.position.entry_time else 0,
            'entry_time': self.position.entry_time.isoformat() if self.position.entry_time else None
        }
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        win_rate = (self.successful_signals / max(1, self.total_signals)) * 100
        
        # Calculate average wins/losses
        all_wins = []
        all_losses = []
        for side_results in self.signal_performance.values():
            for result in side_results:
                if result > 0:
                    all_wins.append(result)
                else:
                    all_losses.append(result)
        
        avg_win = np.mean(all_wins) if all_wins else 0
        avg_loss = np.mean(all_losses) if all_losses else 0
        profit_factor = abs(sum(all_wins) / sum(all_losses)) if all_losses else float('inf')
        
        return {
            'total_signals': self.total_signals,
            'successful_signals': self.successful_signals,
            'win_rate': win_rate,
            'consecutive_losses': self.consecutive_losses,
            'daily_pnl': self.daily_pnl,
            'position_size_multiplier': self.position_size_multiplier,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trades_today': self.trades_today
        }
    
    def get_technical_indicators(self) -> Dict:
        """Get current technical indicators"""
        return self._calculate_technical_indicators()
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        logging.info("📊 Daily trading stats reset")


if __name__ == "__main__":
    # Test enhanced trading logic
    config = {
        'profit_target_ticks': 10,
        'stop_loss_ticks': 5,
        'min_confidence': 0.75
    }
    
    logic = EnhancedBTCTradingLogic(config)
    
    # Sample data
    tick_data = {
        'price': 43250.50,
        'spread': 1.0,
        'size': 0.5,
        'timestamp': datetime.now()
    }
    
    market_analysis = {
        'momentum': 0.004,
        'price_volatility': 0.15,
        'price_change_5': 0.12,
        'price_change_10': 0.25,
        'volume_trend': 0.2,
        'price_above_sma5': True,
        'sma5_above_sma10': True
    }
    
    print("🧪 Testing Enhanced BTC Trading Logic...")
    
    # Add price history
    for i in range(30):
        logic.price_history.append(43000 + i * 5 + np.random.normal(0, 10))
    
    # Test signal generation
    signal = logic.evaluate_tick(tick_data, market_analysis)
    print(f"Signal: {signal.signal_type} | Confidence: {signal.confidence:.2f}")
    print(f"Strength: {signal.strength.value} | Reasoning: {signal.reasoning}")
    
    # Test technical indicators
    indicators = logic.get_technical_indicators()
    print(f"\n📊 Technical Indicators:")
    for key, value in indicators.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    print("✅ Enhanced Trading Logic test completed")