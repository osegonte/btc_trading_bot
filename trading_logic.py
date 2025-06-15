#!/usr/bin/env python3
"""
BTC Swing Trading Logic - €20 to €1M Challenge
Purpose: Implement swing scalping with 2-5 minute holds and market structure awareness
Key Changes: Tick scalping → Swing scalping with percentage-based targets
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from enum import Enum
from collections import deque


class SwingSignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class SwingSignal:
    """Enhanced swing trading signal"""
    def __init__(self, signal_type: SwingSignalType, confidence: float, reasoning: str,
                 entry_price: float = 0.0, target_price: float = 0.0, stop_price: float = 0.0,
                 timeframe: str = "1m", expected_hold_time: int = 180):
        self.signal_type = signal_type
        self.confidence = confidence
        self.reasoning = reasoning
        self.entry_price = entry_price
        self.target_price = target_price
        self.stop_price = stop_price
        self.timeframe = timeframe
        self.expected_hold_time = expected_hold_time  # seconds
        self.timestamp = datetime.now()


class SwingPosition:
    """Manage swing trading position"""
    def __init__(self):
        self.side = None  # 'long', 'short', None
        self.entry_price = None
        self.entry_time = None
        self.quantity = 0.0
        self.target_price = None
        self.stop_price = None
        self.trailing_stop = None
        self.current_balance = 20.0
        self.max_profit = 0.0  # Track peak profit for trailing


class BTCSwingLogic:
    """
    BTC Swing Trading Logic for €20 to €1M Challenge
    Enhanced for 2-5 minute swing positions with market structure awareness
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Swing Trading Configuration
        self.profit_target_pct = config.get('profit_target_pct', 2.5)  # 2.5% profit target
        self.stop_loss_pct = config.get('stop_loss_pct', 1.0)          # 1.0% stop loss
        self.min_confidence = config.get('min_confidence', 0.65)       # Higher confidence for swings
        self.max_position_time = config.get('max_position_time', 300)  # 5 minutes max
        self.min_position_time = config.get('min_position_time', 120)  # 2 minutes min
        self.risk_per_trade_pct = config.get('risk_per_trade_pct', 1.5) # 1.5% risk per trade
        self.position_multiplier = config.get('position_multiplier', 1.5) # 1.5x sustainable
        
        # Position management
        self.position = SwingPosition()
        
        # Challenge tracking with level system
        self.current_balance = 20.0
        self.challenge_level = 0
        self.level_targets = self._generate_level_targets()
        self.daily_loss_limit_pct = config.get('daily_loss_limit_pct', 10.0)
        self.force_reset_balance = config.get('force_reset_balance', 5.0)
        
        # ML Interface
        self.ml_interface = None
        self.current_trade_features = {}
        
        # Swing trading metrics
        self.trades_today = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.signal_cooldown = 30  # 30 seconds between signals
        
        # Technical analysis for swings
        self.candle_buffer_1m = deque(maxlen=50)
        self.candle_buffer_3m = deque(maxlen=20)
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_pnl = 0.0
        self.session_start_balance = 20.0
        
        logging.info(f"✅ BTC Swing Logic initialized")
        logging.info(f"   🎯 Target: {self.profit_target_pct}% | Stop: {self.stop_loss_pct}%")
        logging.info(f"   ⏱️ Hold time: {self.min_position_time}-{self.max_position_time}s")
        logging.info(f"   💰 Starting balance: €{self.current_balance}")
        logging.info(f"   🔄 Position multiplier: {self.position_multiplier}x")
    
    def _generate_level_targets(self) -> list:
        """Generate €20 to €1M level targets"""
        targets = []
        current = 20.0
        while current < 1000000:
            current *= 2
            targets.append(current)
        return targets
    
    def set_ml_interface(self, ml_interface):
        """Set ML interface for enhanced signal generation"""
        self.ml_interface = ml_interface
        logging.info("🤖 ML interface connected for swing trading enhancement")
    
    def can_trade(self) -> tuple[bool, str]:
        """Check if swing trading is allowed"""
        
        # Check if already in position
        if self.position.side:
            return False, f"Already in {self.position.side} swing position"
        
        # Check minimum time between trades
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < self.signal_cooldown:
                return False, f"Signal cooldown active ({time_since_last:.0f}s)"
        
        # Check consecutive losses (swing trading should be more forgiving)
        if self.consecutive_losses >= 3:
            return False, f"Too many consecutive losses: {self.consecutive_losses}"
        
        # Check daily loss limit
        daily_loss_limit = self.current_balance * (self.daily_loss_limit_pct / 100)
        if self.daily_pnl < -daily_loss_limit:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"
        
        # Check minimum balance for swing trading
        min_swing_balance = 15.0
        if self.current_balance < min_swing_balance:
            return False, f"Insufficient balance for swing: €{self.current_balance:.2f}"
        
        return True, "OK"
    
    def evaluate_candle(self, candle_data: Dict, swing_metrics: Dict) -> SwingSignal:
        """
        Main evaluation method for swing trading signals
        Processes completed candles instead of individual ticks
        """
        
        # Update candle buffers
        self._update_candle_buffers(candle_data)
        
        # Check exit conditions first if in position
        if self.position.side:
            return self._check_swing_exit_conditions(candle_data, swing_metrics)
        
        # Check entry conditions if no position
        can_trade, reason = self.can_trade()
        if not can_trade:
            return SwingSignal(SwingSignalType.HOLD, 0.0, reason)
        
        # Analyze swing entry opportunities
        return self._analyze_swing_entry(candle_data, swing_metrics)
    
    def _update_candle_buffers(self, candle_data: Dict):
        """Update candle buffers for analysis"""
        if candle_data['timeframe'] == '1m':
            self.candle_buffer_1m.append(candle_data)
        elif candle_data['timeframe'] == '3m':
            self.candle_buffer_3m.append(candle_data)
    
    def _analyze_swing_entry(self, candle_data: Dict, swing_metrics: Dict) -> SwingSignal:
        """Analyze swing trading entry opportunities with market structure"""
        
        if len(self.candle_buffer_1m) < 20:
            return SwingSignal(SwingSignalType.HOLD, 0.0, "Insufficient candle data for swing analysis")
        
        current_price = candle_data['close']
        timeframe = candle_data['timeframe']
        
        # Only generate signals on 1m candle completions (3m for confirmation)
        if timeframe != '1m':
            return SwingSignal(SwingSignalType.HOLD, 0.0, "Waiting for 1m candle completion")
        
        # Extract swing metrics
        trend_direction = swing_metrics.get('trend_direction', 'neutral')
        ma_alignment = swing_metrics.get('ma_alignment', {})
        momentum_1m = swing_metrics.get('momentum_1m', 0)
        momentum_3m = swing_metrics.get('momentum_3m', 0)
        current_rsi = swing_metrics.get('current_rsi', 50)
        atr = swing_metrics.get('atr', 0)
        support_levels = swing_metrics.get('support_levels', [])
        resistance_levels = swing_metrics.get('resistance_levels', [])
        volume_surge = swing_metrics.get('volume_surge', False)
        vwap_position = swing_metrics.get('vwap_position', 'neutral')
        
        # Store features for ML learning
        self.current_trade_features = {
            'current_price': current_price,
            'trend_direction': trend_direction,
            'ma_aligned': ma_alignment.get('aligned', False),
            'ma_direction': ma_alignment.get('direction', 'neutral'),
            'momentum_1m': momentum_1m,
            'momentum_3m': momentum_3m,
            'rsi': current_rsi,
            'atr': atr,
            'volume_surge': volume_surge,
            'vwap_position': vwap_position,
            'balance': self.current_balance,
            'consecutive_losses': self.consecutive_losses,
            'body_size': candle_data.get('body_size', 0),
            'is_bullish_candle': candle_data.get('is_bullish', False),
            'candle_range': candle_data.get('range', 0),
            'near_support': self._near_level(current_price, support_levels),
            'near_resistance': self._near_level(current_price, resistance_levels),
            'swing_setup': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get ML enhancement if available
        ml_signal = None
        if self.ml_interface:
            try:
                # Create tick-like data for ML compatibility
                tick_like_data = {
                    'price': current_price,
                    'size': candle_data.get('volume', 1.0),
                    'timestamp': datetime.now()
                }
                ml_signal = self.ml_interface.process_tick(tick_like_data)
                if ml_signal.signal != 'hold':
                    logging.debug(f"🤖 ML Swing Signal: {ml_signal.signal} (conf: {ml_signal.confidence:.2f})")
            except Exception as e:
                logging.warning(f"ML processing error: {e}")
        
        # SWING SIGNAL SCORING SYSTEM
        bullish_score = 0
        bearish_score = 0
        
        # 1. Trend Direction (higher weight for swings)
        if trend_direction == 'uptrend':
            bullish_score += 4
        elif trend_direction == 'downtrend':
            bearish_score += 4
        
        # 2. Moving Average Alignment
        if ma_alignment.get('aligned', False):
            if ma_alignment.get('direction') == 'bullish':
                bullish_score += 3
            elif ma_alignment.get('direction') == 'bearish':
                bearish_score += 3
        
        # 3. Multi-timeframe Momentum Confirmation
        if momentum_1m > 0.002 and momentum_3m > 0.001:  # Strong bullish momentum
            bullish_score += 3
        elif momentum_1m > 0.001:  # Moderate bullish momentum
            bullish_score += 2
        elif momentum_1m < -0.002 and momentum_3m < -0.001:  # Strong bearish momentum
            bearish_score += 3
        elif momentum_1m < -0.001:  # Moderate bearish momentum
            bearish_score += 2
        
        # 4. RSI Conditions for Swing Trading (25-75 range)
        if 25 < current_rsi < 50:
            bullish_score += 2
        elif 50 < current_rsi < 75:
            bearish_score += 2
        elif current_rsi < 25:  # Oversold - potential bounce
            bullish_score += 3
        elif current_rsi > 75:  # Overbought - potential drop
            bearish_score += 3
        
        # 5. Support/Resistance Interaction
        if self._near_level(current_price, support_levels, tolerance=0.005):
            bullish_score += 2  # Bounce off support
        if self._near_level(current_price, resistance_levels, tolerance=0.005):
            bearish_score += 2  # Rejection at resistance
        
        # 6. Volume Confirmation
        if volume_surge:
            if momentum_1m > 0:
                bullish_score += 2
            elif momentum_1m < 0:
                bearish_score += 2
        
        # 7. VWAP Position
        if vwap_position == 'above' and momentum_1m > 0:
            bullish_score += 1
        elif vwap_position == 'below' and momentum_1m < 0:
            bearish_score += 1
        
        # 8. Candle Pattern Analysis
        candle_patterns = self._analyze_candle_patterns()
        if candle_patterns.get('bullish_pattern'):
            bullish_score += 2
        if candle_patterns.get('bearish_pattern'):
            bearish_score += 2
        
        # 9. ML Enhancement (significant boost for swings)
        if ml_signal:
            if ml_signal.signal == 'buy' and ml_signal.confidence > 0.6:
                bullish_score += 3
                logging.debug("🤖 ML enhancing bullish swing signal")
            elif ml_signal.signal == 'sell' and ml_signal.confidence > 0.6:
                bearish_score += 3
                logging.debug("🤖 ML enhancing bearish swing signal")
        
        # Volatility check - ensure sufficient movement potential
        if atr < 10:  # Too quiet for swing trading
            return SwingSignal(SwingSignalType.HOLD, 0.0, "Insufficient volatility for swing trading")
        elif atr > 200:  # Too volatile - risky
            return SwingSignal(SwingSignalType.HOLD, 0.0, "Excessive volatility - too risky for swing")
        
        # Generate swing signal
        signal_type = SwingSignalType.HOLD
        confidence = 0.0
        reasoning = ""
        
        # Require higher scores for swing trading (more selective)
        if bullish_score >= 8 and bullish_score > bearish_score:
            signal_type = SwingSignalType.BUY
            confidence = min(0.95, 0.5 + (bullish_score / 20))
            reasoning = f"Bullish swing setup: Score {bullish_score}"
            if ml_signal and ml_signal.signal == 'buy':
                reasoning += f" + ML({ml_signal.confidence:.2f})"
                confidence = min(0.95, confidence + 0.1)
        
        elif bearish_score >= 8 and bearish_score > bullish_score:
            signal_type = SwingSignalType.SELL
            confidence = min(0.95, 0.5 + (bearish_score / 20))
            reasoning = f"Bearish swing setup: Score {bearish_score}"
            if ml_signal and ml_signal.signal == 'sell':
                reasoning += f" + ML({ml_signal.confidence:.2f})"
                confidence = min(0.95, confidence + 0.1)
        
        else:
            max_score = max(bullish_score, bearish_score)
            confidence = max_score / 20
            reasoning = f"Mixed signals: Bull {bullish_score}, Bear {bearish_score}"
        
        # Apply confidence filter
        if confidence < self.min_confidence:
            return SwingSignal(SwingSignalType.HOLD, confidence, 
                             f"Low confidence for swing: {confidence:.2f}")
        
        # Calculate swing targets if signal generated
        if signal_type in [SwingSignalType.BUY, SwingSignalType.SELL]:
            entry_price = current_price
            
            # Calculate percentage-based targets
            if signal_type == SwingSignalType.BUY:
                target_price = entry_price * (1 + self.profit_target_pct / 100)
                stop_price = entry_price * (1 - self.stop_loss_pct / 100)
            else:  # SELL
                target_price = entry_price * (1 - self.profit_target_pct / 100)
                stop_price = entry_price * (1 + self.stop_loss_pct / 100)
            
            # Estimate hold time based on volatility and momentum
            expected_hold = self._estimate_hold_time(atr, abs(momentum_1m))
            
            return SwingSignal(
                signal_type, confidence, reasoning,
                entry_price, target_price, stop_price,
                timeframe='1m', expected_hold_time=expected_hold
            )
        
        return SwingSignal(SwingSignalType.HOLD, confidence, reasoning)
    
    def _near_level(self, price: float, levels: list, tolerance: float = 0.003) -> bool:
        """Check if price is near support/resistance level"""
        for level in levels[:3]:  # Check top 3 levels
            if abs(price - level) / level < tolerance:
                return True
        return False
    
    def _analyze_candle_patterns(self) -> Dict:
        """Analyze recent candle patterns for swing signals"""
        if len(self.candle_buffer_1m) < 3:
            return {}
        
        recent_candles = list(self.candle_buffer_1m)[-3:]
        
        # Simple pattern recognition
        patterns = {
            'bullish_pattern': False,
            'bearish_pattern': False
        }
        
        # Three white soldiers / three black crows (simplified)
        all_bullish = all(candle.get('is_bullish', False) for candle in recent_candles)
        all_bearish = all(candle.get('is_bearish', False) for candle in recent_candles)
        
        if all_bullish:
            patterns['bullish_pattern'] = True
        elif all_bearish:
            patterns['bearish_pattern'] = True
        
        # Doji reversal (simplified)
        latest = recent_candles[-1]
        if latest.get('body_size', 0) < latest.get('range', 1) * 0.1:  # Small body
            if len(recent_candles) >= 2:
                prev = recent_candles[-2]
                if prev.get('is_bullish') and latest.get('range') > prev.get('range', 0):
                    patterns['bearish_pattern'] = True
                elif prev.get('is_bearish') and latest.get('range') > prev.get('range', 0):
                    patterns['bullish_pattern'] = True
        
        return patterns
    
    def _estimate_hold_time(self, atr: float, momentum: float) -> int:
        """Estimate expected hold time based on market conditions"""
        base_time = 180  # 3 minutes base
        
        # Adjust based on volatility
        if atr > 50:
            base_time = 120  # Faster moves in volatile market
        elif atr < 20:
            base_time = 240  # Slower moves in quiet market
        
        # Adjust based on momentum
        if momentum > 0.003:
            base_time = max(120, base_time - 60)  # Strong momentum = faster
        elif momentum < 0.001:
            base_time = min(300, base_time + 60)  # Weak momentum = slower
        
        return base_time
    
    def _check_swing_exit_conditions(self, candle_data: Dict, swing_metrics: Dict) -> SwingSignal:
        """Check exit conditions for current swing position"""
        
        if not self.position.side:
            return SwingSignal(SwingSignalType.HOLD, 0.0, "No position to exit")
        
        current_price = candle_data['close']
        current_time = datetime.now()
        
        # Calculate current P&L percentage
        if self.position.side == 'long':
            pnl_pct = ((current_price - self.position.entry_price) / self.position.entry_price) * 100
        else:  # short
            pnl_pct = ((self.position.entry_price - current_price) / self.position.entry_price) * 100
        
        # Time-based exits
        time_in_position = (current_time - self.position.entry_time).total_seconds()
        
        # Maximum time exit
        if time_in_position > self.max_position_time:
            return SwingSignal(
                SwingSignalType.CLOSE, 1.0,
                f"Max time exit: {time_in_position:.0f}s (max {self.max_position_time}s)"
            )
        
        # Minimum time protection (avoid premature exits)
        if time_in_position < self.min_position_time:
            # Only allow exits for stop loss if within minimum time
            if pnl_pct <= -self.stop_loss_pct:
                return SwingSignal(
                    SwingSignalType.CLOSE, 1.0,
                    f"Stop loss hit early: {pnl_pct:.2f}%"
                )
            else:
                return SwingSignal(
                    SwingSignalType.HOLD, 0.0,
                    f"Minimum hold time: {time_in_position:.0f}s/{self.min_position_time}s"
                )
        
        # Profit target hit
        if pnl_pct >= self.profit_target_pct:
            return SwingSignal(
                SwingSignalType.CLOSE, 1.0,
                f"Profit target hit: +{pnl_pct:.2f}%"
            )
        
        # Stop loss hit
        if pnl_pct <= -self.stop_loss_pct:
            return SwingSignal(
                SwingSignalType.CLOSE, 1.0,
                f"Stop loss hit: {pnl_pct:.2f}%"
            )
        
        # Trailing stop logic
        if pnl_pct > 0:
            self.position.max_profit = max(self.position.max_profit, pnl_pct)
            
            # Activate trailing stop at 50% of target
            if self.position.max_profit >= self.profit_target_pct * 0.5:
                trailing_stop_pct = self.stop_loss_pct * 0.5  # Tighter trailing stop
                
                if self.position.max_profit - pnl_pct >= trailing_stop_pct:
                    return SwingSignal(
                        SwingSignalType.CLOSE, 0.9,
                        f"Trailing stop: Peak {self.position.max_profit:.2f}%, Now {pnl_pct:.2f}%"
                    )
        
        # Market structure reversal
        trend_direction = swing_metrics.get('trend_direction', 'neutral')
        if self.position.side == 'long' and trend_direction == 'downtrend':
            if pnl_pct < self.profit_target_pct * 0.3:  # Not enough profit to ignore
                return SwingSignal(
                    SwingSignalType.CLOSE, 0.7,
                    f"Trend reversal against long position"
                )
        elif self.position.side == 'short' and trend_direction == 'uptrend':
            if pnl_pct < self.profit_target_pct * 0.3:
                return SwingSignal(
                    SwingSignalType.CLOSE, 0.7,
                    f"Trend reversal against short position"
                )
        
        # RSI extreme reversal
        current_rsi = swing_metrics.get('current_rsi', 50)
        if self.position.side == 'long' and current_rsi > 80:
            if pnl_pct > 0:  # Take profit in overbought
                return SwingSignal(
                    SwingSignalType.CLOSE, 0.6,
                    f"RSI overbought exit: {current_rsi:.1f}"
                )
        elif self.position.side == 'short' and current_rsi < 20:
            if pnl_pct > 0:  # Take profit in oversold
                return SwingSignal(
                    SwingSignalType.CLOSE, 0.6,
                    f"RSI oversold exit: {current_rsi:.1f}"
                )
        
        # Hold position
        return SwingSignal(
            SwingSignalType.HOLD, 0.0,
            f"Holding swing: {pnl_pct:+.2f}% ({time_in_position:.0f}s)"
        )
    
    def _calculate_position_size(self, current_price: float) -> float:
        """Calculate position size for swing trading with 1.5x multiplier"""
        
        # Risk amount (1.5% of balance with 1.5x multiplier)
        risk_amount = self.current_balance * (self.risk_per_trade_pct / 100) * self.position_multiplier
        
        # Position size based on 1% stop loss
        stop_distance_pct = self.stop_loss_pct / 100
        position_value = risk_amount / stop_distance_pct
        
        # Convert to BTC quantity
        position_size = position_value / current_price
        
        # Apply reasonable limits for swing trading
        min_size = 0.0001  # Minimum 0.0001 BTC
        max_size = 0.01    # Maximum 0.01 BTC per trade
        
        position_size = max(min_size, min(position_size, max_size))
        
        return round(position_size, 6)
    
    def update_position(self, action: str, price: float, quantity: float, timestamp: str = None):
        """Update position state with swing trading enhancements"""
        
        if action in ['buy', 'sell']:
            # Open new swing position
            self.position.side = 'long' if action == 'buy' else 'short'
            self.position.entry_price = price
            self.position.entry_time = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
            self.position.quantity = quantity
            self.position.max_profit = 0.0
            
            # Calculate percentage-based targets
            if action == 'buy':
                self.position.target_price = price * (1 + self.profit_target_pct / 100)
                self.position.stop_price = price * (1 - self.stop_loss_pct / 100)
            else:
                self.position.target_price = price * (1 - self.profit_target_pct / 100)
                self.position.stop_price = price * (1 + self.stop_loss_pct / 100)
            
            self.last_trade_time = datetime.now()
            
            position_value = quantity * price
            logging.info(f"✅ Swing position opened: {self.position.side.upper()} {quantity:.6f} BTC @ €{price:.2f} (€{position_value:.2f})")
            
        elif action == 'close':
            # Close swing position and update performance
            if self.position.side and self.position.entry_price:
                # Calculate final P&L
                if self.position.side == 'long':
                    pnl_pct = ((price - self.position.entry_price) / self.position.entry_price) * 100
                else:
                    pnl_pct = ((self.position.entry_price - price) / self.position.entry_price) * 100
                
                total_pnl = (pnl_pct / 100) * self.current_balance * (self.risk_per_trade_pct / 100) * self.position_multiplier / (self.stop_loss_pct / 100)
                
                # Update balance and tracking
                self.current_balance += total_pnl
                self.daily_pnl += total_pnl
                self.total_trades += 1
                
                # Track wins/losses and ML learning
                if total_pnl > 0:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                    outcome = 'profitable'
                else:
                    self.consecutive_losses += 1
                    outcome = 'unprofitable'
                
                # ML Learning for swing trading
                if self.ml_interface and self.current_trade_features:
                    try:
                        self.current_trade_features['hold_time'] = (datetime.now() - self.position.entry_time).total_seconds()
                        self.current_trade_features['pnl_percentage'] = pnl_pct
                        self.current_trade_features['max_profit_reached'] = self.position.max_profit
                        
                        self.ml_interface.record_trade_outcome(
                            self.current_trade_features,
                            self.position.side,
                            total_pnl
                        )
                        logging.debug(f"🤖 ML learned from swing trade: {outcome} ({pnl_pct:+.2f}%)")
                    except Exception as e:
                        logging.warning(f"ML learning error: {e}")
                
                hold_time = (datetime.now() - self.position.entry_time).total_seconds()
                logging.info(f"💰 Swing position closed: {pnl_pct:+.2f}% (€{total_pnl:+.2f}) | Hold: {hold_time:.0f}s | Balance: €{self.current_balance:.2f}")
                
                # Check level progression
                self._check_level_progression()
                
                # Check if reset needed
                if self.current_balance < self.force_reset_balance:
                    self._reset_challenge()
            
            # Reset position
            self.position = SwingPosition()
            self.position.current_balance = self.current_balance
            self.current_trade_features = {}
            self.last_trade_time = datetime.now()
    
    def _check_level_progression(self):
        """Check if reached next challenge level"""
        current_level = 0
        for i, target in enumerate(self.level_targets):
            if self.current_balance >= target:
                current_level = i + 1
            else:
                break
        
        if current_level > self.challenge_level:
            self.challenge_level = current_level
            target_reached = self.level_targets[current_level - 1]
            logging.info(f"🎉 SWING LEVEL {current_level} REACHED! €{target_reached:.0f}")
            
            if target_reached >= 1000000:
                logging.info("🏆 SWING CHALLENGE COMPLETED! €1,000,000 REACHED!")
    
    def _reset_challenge(self):
        """Reset challenge to €20 with enhanced tracking"""
        logging.info(f"🔄 SWING CHALLENGE RESET")
        logging.info(f"   Previous balance: €{self.current_balance:.2f}")
        logging.info(f"   Level reached: {self.challenge_level}")
        logging.info(f"   Trades completed: {self.total_trades}")
        
        # Reset financial metrics
        self.current_balance = 20.0
        self.session_start_balance = 20.0
        self.challenge_level = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trades_today = 0
        
        # Reset position
        self.position = SwingPosition()
        self.position.current_balance = 20.0
        
        logging.info("✅ Swing challenge reset to €20")
    
    def get_position_info(self) -> Dict:
        """Get current swing position information"""
        if not self.position.side:
            return {
                'has_position': False,
                'balance': self.current_balance,
                'trades_today': self.trades_today,
                'consecutive_losses': self.consecutive_losses,
                'challenge_level': self.challenge_level,
                'swing_mode': True
            }
        
        time_in_position = (datetime.now() - self.position.entry_time).total_seconds()
        current_price = self.position.entry_price  # Simplified - would use real current price
        
        if self.position.side == 'long':
            pnl_pct = ((current_price - self.position.entry_price) / self.position.entry_price) * 100
        else:
            pnl_pct = ((self.position.entry_price - current_price) / self.position.entry_price) * 100
        
        return {
            'has_position': True,
            'side': self.position.side,
            'entry_price': self.position.entry_price,
            'quantity': self.position.quantity,
            'target_price': self.position.target_price,
            'stop_price': self.position.stop_price,
            'time_in_position': time_in_position,
            'entry_time': self.position.entry_time.isoformat(),
            'current_pnl_pct': pnl_pct,
            'max_profit': self.position.max_profit,
            'balance': self.current_balance,
            'challenge_level': self.challenge_level,
            'swing_mode': True
        }
    
    def get_swing_performance(self) -> Dict:
        """Get comprehensive swing trading performance metrics"""
        
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        balance_growth = ((self.current_balance - self.session_start_balance) / self.session_start_balance) * 100
        
        # Calculate next target
        next_target = 1000000
        for target in self.level_targets:
            if self.current_balance < target:
                next_target = target
                break
        
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
            
            # Challenge specific metrics
            'challenge_level': self.challenge_level,
            'next_target': next_target,
            'progress_to_next_pct': progress_to_next,
            'distance_to_million': 1000000 - self.current_balance,
            
            # Swing trading specific
            'profit_target_pct': self.profit_target_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'position_multiplier': self.position_multiplier,
            'min_hold_time': self.min_position_time,
            'max_hold_time': self.max_position_time,
            'trading_mode': 'swing_scalping'
        }
    
    def get_risk_metrics(self) -> Dict:
        """Get risk management metrics for swing trading"""
        
        max_risk_per_trade = self.current_balance * (self.risk_per_trade_pct / 100) * self.position_multiplier
        daily_loss_limit = self.current_balance * (self.daily_loss_limit_pct / 100)
        daily_risk_utilization = (abs(self.daily_pnl) / daily_loss_limit) * 100 if self.daily_pnl < 0 else 0
        
        return {
            'current_balance': self.current_balance,
            'risk_per_trade_euros': max_risk_per_trade,
            'risk_per_trade_pct': self.risk_per_trade_pct,
            'position_multiplier': self.position_multiplier,
            'daily_loss_limit': daily_loss_limit,
            'daily_pnl': self.daily_pnl,
            'daily_risk_utilization_pct': daily_risk_utilization,
            'consecutive_losses': self.consecutive_losses,
            'force_reset_threshold': self.force_reset_balance,
            'should_reset': self.current_balance < self.force_reset_balance,
            'profit_target_pct': self.profit_target_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'min_confidence': self.min_confidence,
            'signal_cooldown': self.signal_cooldown,
            'swing_trading_mode': True
        }
    
    def reset_to_twenty_euros(self):
        """Manual reset to €20 for new attempt"""
        self._reset_challenge()
    
    def should_reset_challenge(self) -> bool:
        """Check if challenge should be reset"""
        return self.current_balance < self.force_reset_balance


if __name__ == "__main__":
    # Test BTC swing trading logic
    config = {
        'profit_target_pct': 2.5,
        'stop_loss_pct': 1.0,
        'min_confidence': 0.65,
        'max_position_time': 300,
        'min_position_time': 120,
        'position_multiplier': 1.5
    }
    
    logic = BTCSwingLogic(config)
    
    print("🧪 Testing BTC Swing Trading Logic...")
    
    # Test position size calculation
    test_prices = [43000, 50000, 40000]
    test_balances = [20, 40, 80, 160, 320, 640]
    
    print("\n💰 SWING POSITION SIZE CALCULATION:")
    print("=" * 60)
    
    for balance in test_balances:
        logic.current_balance = balance
        for price in test_prices[:1]:  # Test with first price
            pos_size = logic._calculate_position_size(price)
            pos_value = pos_size * price
            risk_amount = balance * (logic.risk_per_trade_pct / 100) * logic.position_multiplier
            
            print(f"   €{balance:3.0f} account → {pos_size:.6f} BTC = €{pos_value:6.2f} | Risk: €{risk_amount:5.2f}")
    
    # Test swing performance metrics
    print(f"\n📊 SWING PERFORMANCE METRICS:")
    print("=" * 50)
    
    logic.current_balance = 20.0
    performance = logic.get_swing_performance()
    risk_metrics = logic.get_risk_metrics()
    
    for key, value in performance.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value}")
    
    print(f"\n⚖️ SWING RISK METRICS:")
    for key, value in risk_metrics.items():
        if isinstance(value, (int, float, bool)):
            print(f"   {key}: {value}")
    
    # Test mock swing signal generation
    print(f"\n🎯 MOCK SWING SIGNAL TEST:")
    print("=" * 40)
    
    mock_candle = {
        'timeframe': '1m',
        'timestamp': datetime.now(),
        'open': 43000,
        'high': 43050,
        'low': 42980,
        'close': 43030,
        'volume': 1.5,
        'body_size': 30,
        'is_bullish': True,
        'range': 70
    }
    
    mock_metrics = {
        'trend_direction': 'uptrend',
        'ma_alignment': {'aligned': True, 'direction': 'bullish'},
        'momentum_1m': 0.003,
        'momentum_3m': 0.002,
        'current_rsi': 45,
        'atr': 25,
        'support_levels': [42900, 42800],
        'resistance_levels': [43100, 43200],
        'volume_surge': True,
        'vwap_position': 'above'
    }
    
    signal = logic._analyze_swing_entry(mock_candle, mock_metrics)
    
    print(f"Signal Type: {signal.signal_type.value}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Reasoning: {signal.reasoning}")
    if signal.signal_type != SwingSignalType.HOLD:
        print(f"Entry: €{signal.entry_price:.2f}")
        print(f"Target: €{signal.target_price:.2f} ({signal.profit_target_pct:.1f}%)")
        print(f"Stop: €{signal.stop_price:.2f} ({signal.stop_loss_pct:.1f}%)")
        print(f"Expected Hold: {signal.expected_hold_time}s")
    
    print("\n✅ BTC SWING TRADING LOGIC READY!")
    print("=" * 50)
    print("✅ Percentage-based targets: 2.5% profit, 1% stop")
    print("✅ Hold times: 2-5 minutes")
    print("✅ Market structure awareness")
    print("✅ Multi-timeframe confirmation")
    print("✅ Enhanced risk management")
    print("✅ ML learning integration")
    print("✅ €20 to €1M challenge tracking")