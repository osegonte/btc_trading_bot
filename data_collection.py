#!/usr/bin/env python3
"""
BTC Swing Data Collection - Enhanced for 2-5 minute swings
Purpose: Build candles from ticks and analyze market structure for swing trading
Key Changes: Tick scalping → Candle-based swing analysis
"""

import time
import logging
import threading
import numpy as np
import json
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass


@dataclass
class BTCCandle:
    """1-minute or 3-minute BTC candle for swing analysis"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    timeframe: str  # '1m' or '3m'
    
    def __post_init__(self):
        # Calculate additional swing trading metrics
        self.body_size = abs(self.close - self.open)
        self.upper_wick = self.high - max(self.open, self.close)
        self.lower_wick = min(self.open, self.close) - self.low
        self.range = self.high - self.low
        self.is_bullish = self.close > self.open
        self.is_bearish = self.close < self.open


class MarketStructure:
    """Analyze BTC market structure for swing trading"""
    def __init__(self):
        self.swing_highs = deque(maxlen=20)
        self.swing_lows = deque(maxlen=20)
        self.support_levels = deque(maxlen=10)
        self.resistance_levels = deque(maxlen=10)
        self.trend_direction = "neutral"  # "uptrend", "downtrend", "neutral"
        self.last_structure_update = None
    
    def update_structure(self, candles: List[BTCCandle]):
        """Update market structure from recent candles"""
        if len(candles) < 10:
            return
        
        # Find swing highs and lows (simplified pivot detection)
        for i in range(2, len(candles) - 2):
            current = candles[i]
            
            # Swing high: higher than 2 candles before and after
            if (current.high > candles[i-1].high and current.high > candles[i-2].high and 
                current.high > candles[i+1].high and current.high > candles[i+2].high):
                self.swing_highs.append((current.timestamp, current.high))
                self.resistance_levels.append(current.high)
            
            # Swing low: lower than 2 candles before and after
            if (current.low < candles[i-1].low and current.low < candles[i-2].low and 
                current.low < candles[i+1].low and current.low < candles[i+2].low):
                self.swing_lows.append((current.timestamp, current.low))
                self.support_levels.append(current.low)
        
        # Determine trend direction
        if len(self.swing_highs) >= 2 and len(self.swing_lows) >= 2:
            recent_highs = list(self.swing_highs)[-2:]
            recent_lows = list(self.swing_lows)[-2:]
            
            higher_highs = recent_highs[1][1] > recent_highs[0][1]
            higher_lows = recent_lows[1][1] > recent_lows[0][1]
            lower_highs = recent_highs[1][1] < recent_highs[0][1]
            lower_lows = recent_lows[1][1] < recent_lows[0][1]
            
            if higher_highs and higher_lows:
                self.trend_direction = "uptrend"
            elif lower_highs and lower_lows:
                self.trend_direction = "downtrend"
            else:
                self.trend_direction = "neutral"
        
        self.last_structure_update = datetime.now()


class BTCSwingDataCollector:
    """
    Enhanced BTC data collector for swing trading
    Builds 1m and 3m candles from tick data with market structure analysis
    """
    
    def __init__(self, symbol: str = "BTCUSD"):
        self.symbol = symbol
        self.is_running = False
        
        # Candle building for swing trading
        self.current_1m_candle = None
        self.current_3m_candle = None
        self.candles_1m = deque(maxlen=100)  # Keep 100 1-minute candles
        self.candles_3m = deque(maxlen=50)   # Keep 50 3-minute candles
        
        # Candle completion callbacks
        self.candle_callbacks = []
        
        # Raw tick data (still needed for candle building)
        self.tick_buffer = deque(maxlen=50)
        self.current_price = 43000.0
        self.tick_count = 0
        
        # Market structure analysis
        self.market_structure = MarketStructure()
        
        # Technical indicators
        self.sma_9_1m = deque(maxlen=50)
        self.sma_20_1m = deque(maxlen=50)
        self.ema_12_1m = deque(maxlen=50)
        self.ema_26_1m = deque(maxlen=50)
        self.rsi_14_1m = deque(maxlen=50)
        
        # Volume analysis
        self.volume_profile = {}
        self.vwap_anchor = None
        
        # Connection management
        self.data_source = "none"
        self.connection_stable = False
        
        logging.info(f"✅ BTC Swing Data Collector initialized for {symbol}")
    
    def add_candle_callback(self, callback: Callable):
        """Add callback for candle completion"""
        self.candle_callbacks.append(callback)
        logging.debug(f"Added candle callback, total: {len(self.candle_callbacks)}")
    
    def start_data_feed(self, api_key: str = "", secret_key: str = ""):
        """Start BTC data feed optimized for swing trading"""
        self.is_running = True
        logging.info("🚀 Starting BTC swing trading data feed...")
        
        # Try Alpaca API first
        if api_key and secret_key and api_key != 'YOUR_ALPACA_API_KEY':
            try:
                self._start_alpaca_feed(api_key, secret_key)
                time.sleep(2)
                if self.tick_count > 0:
                    logging.info("✅ Alpaca BTC data feed active")
                    return
            except Exception as e:
                logging.warning(f"❌ Alpaca connection failed: {e}")
        
        # Fallback to swing trading simulation
        logging.info("🎮 Starting BTC swing simulation...")
        self._start_swing_simulation()
    
    def _start_alpaca_feed(self, api_key: str, secret_key: str):
        """Start Alpaca crypto data feed"""
        try:
            import alpaca_trade_api as tradeapi
            
            self.alpaca_api = tradeapi.StreamConn(
                api_key, 
                secret_key,
                base_url='https://paper-api.alpaca.markets',
                data_feed='crypto'
            )
            
            @self.alpaca_api.on(f'^T\\.{self.symbol}')
            async def on_btc_tick(conn, channel, data):
                """Handle incoming BTC tick from Alpaca"""
                self._process_tick_for_candles(
                    price=float(data.price),
                    volume=float(data.size),
                    timestamp=datetime.now()
                )
            
            def run_alpaca_stream():
                try:
                    self.alpaca_api.run()
                except Exception as e:
                    logging.error(f"Alpaca stream error: {e}")
                    if self.is_running:
                        self._start_swing_simulation()
            
            alpaca_thread = threading.Thread(target=run_alpaca_stream, daemon=True)
            alpaca_thread.start()
            self.data_source = "alpaca"
            self.connection_stable = True
            
        except ImportError:
            raise Exception("alpaca-trade-api not installed")
        except Exception as e:
            raise Exception(f"Alpaca setup failed: {e}")
    
    def _start_swing_simulation(self):
        """Start simulation optimized for swing trading patterns"""
        
        def simulate_btc_swing_patterns():
            """Generate realistic BTC price action for swing trading"""
            base_price = 43000.0
            trend_strength = 0.0
            trend_duration = 0
            swing_phase = "accumulation"  # accumulation, breakout, trend, reversal
            phase_duration = 0
            
            logging.info(f"🎮 BTC swing simulation started at ${base_price:,.2f}")
            
            while self.is_running:
                try:
                    current_time = datetime.now()
                    
                    # Swing phase management
                    if phase_duration <= 0:
                        # Change swing phase
                        phases = ["accumulation", "breakout", "trend", "reversal"]
                        swing_phase = np.random.choice(phases)
                        
                        if swing_phase == "accumulation":
                            phase_duration = np.random.randint(30, 90)  # 30-90 ticks
                            trend_strength = np.random.uniform(-0.2, 0.2)
                        elif swing_phase == "breakout":
                            phase_duration = np.random.randint(15, 30)  # 15-30 ticks
                            trend_strength = np.random.choice([-0.8, 0.8])
                        elif swing_phase == "trend":
                            phase_duration = np.random.randint(60, 180)  # 60-180 ticks
                            trend_strength = np.random.uniform(-0.6, 0.6)
                        else:  # reversal
                            phase_duration = np.random.randint(20, 40)  # 20-40 ticks
                            trend_strength *= -0.7  # Reverse with some dampening
                    
                    # Price movement based on swing phase
                    if swing_phase == "accumulation":
                        # Tight range, low volatility
                        price_move = np.random.normal(0, 5) + trend_strength * 0.5
                        volume_multiplier = np.random.uniform(0.3, 0.8)
                        
                    elif swing_phase == "breakout":
                        # Sharp directional move with volume
                        direction = 1 if trend_strength > 0 else -1
                        price_move = direction * np.random.uniform(15, 40) + np.random.normal(0, 3)
                        volume_multiplier = np.random.uniform(2.0, 4.0)
                        
                    elif swing_phase == "trend":
                        # Sustained directional movement
                        trend_move = trend_strength * np.random.uniform(2, 8)
                        noise = np.random.normal(0, 8)
                        price_move = trend_move + noise
                        volume_multiplier = np.random.uniform(0.8, 1.5)
                        
                    else:  # reversal
                        # Choppy price action with increasing volume
                        price_move = trend_strength * 3 + np.random.normal(0, 12)
                        volume_multiplier = np.random.uniform(1.2, 2.5)
                    
                    # Apply price movement
                    base_price += price_move
                    base_price = max(30000, min(60000, base_price))  # Keep realistic range
                    
                    # Generate volume based on phase
                    base_volume = 0.5
                    volume = base_volume * volume_multiplier * np.random.uniform(0.5, 2.0)
                    
                    # Process tick for candle building
                    self._process_tick_for_candles(base_price, volume, current_time)
                    
                    phase_duration -= 1
                    
                    # Slower tick rate for swing trading (2-3 ticks per second)
                    time.sleep(np.random.uniform(0.3, 0.5))
                    
                except Exception as e:
                    logging.error(f"Swing simulation error: {e}")
                    time.sleep(1)
        
        sim_thread = threading.Thread(target=simulate_btc_swing_patterns, daemon=True)
        sim_thread.start()
        self.data_source = "swing_simulation"
        self.connection_stable = True
    
    def _process_tick_for_candles(self, price: float, volume: float, timestamp: datetime):
        """Process tick and build 1m and 3m candles"""
        
        # Store raw tick
        self.tick_buffer.append({
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        })
        self.current_price = price
        self.tick_count += 1
        
        # Build 1-minute candles
        self._build_1m_candle(price, volume, timestamp)
        
        # Build 3-minute candles
        self._build_3m_candle(price, volume, timestamp)
    
    def _build_1m_candle(self, price: float, volume: float, timestamp: datetime):
        """Build 1-minute candles for swing analysis"""
        
        # Get current minute
        current_minute = timestamp.replace(second=0, microsecond=0)
        
        # Initialize new candle if needed
        if not self.current_1m_candle or self.current_1m_candle.timestamp != current_minute:
            # Complete previous candle
            if self.current_1m_candle:
                self._complete_candle(self.current_1m_candle, '1m')
            
            # Start new candle
            self.current_1m_candle = BTCCandle(
                timestamp=current_minute,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                vwap=price,
                timeframe='1m'
            )
        else:
            # Update existing candle
            self.current_1m_candle.high = max(self.current_1m_candle.high, price)
            self.current_1m_candle.low = min(self.current_1m_candle.low, price)
            self.current_1m_candle.close = price
            self.current_1m_candle.volume += volume
            
            # Update VWAP
            total_value = (self.current_1m_candle.vwap * 
                          (self.current_1m_candle.volume - volume)) + (price * volume)
            self.current_1m_candle.vwap = total_value / self.current_1m_candle.volume
    
    def _build_3m_candle(self, price: float, volume: float, timestamp: datetime):
        """Build 3-minute candles for higher timeframe analysis"""
        
        # Get current 3-minute period (0, 3, 6, 9, etc.)
        minute = timestamp.minute
        period_start_minute = (minute // 3) * 3
        current_3m = timestamp.replace(minute=period_start_minute, second=0, microsecond=0)
        
        # Initialize new candle if needed
        if not self.current_3m_candle or self.current_3m_candle.timestamp != current_3m:
            # Complete previous candle
            if self.current_3m_candle:
                self._complete_candle(self.current_3m_candle, '3m')
            
            # Start new candle
            self.current_3m_candle = BTCCandle(
                timestamp=current_3m,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                vwap=price,
                timeframe='3m'
            )
        else:
            # Update existing candle
            self.current_3m_candle.high = max(self.current_3m_candle.high, price)
            self.current_3m_candle.low = min(self.current_3m_candle.low, price)
            self.current_3m_candle.close = price
            self.current_3m_candle.volume += volume
            
            # Update VWAP
            total_value = (self.current_3m_candle.vwap * 
                          (self.current_3m_candle.volume - volume)) + (price * volume)
            self.current_3m_candle.vwap = total_value / self.current_3m_candle.volume
    
    def _complete_candle(self, candle: BTCCandle, timeframe: str):
        """Complete candle and update indicators"""
        
        # Store completed candle
        if timeframe == '1m':
            self.candles_1m.append(candle)
            self._update_1m_indicators(candle)
        else:  # 3m
            self.candles_3m.append(candle)
        
        # Update market structure
        if timeframe == '1m' and len(self.candles_1m) >= 10:
            self.market_structure.update_structure(list(self.candles_1m)[-10:])
        
        # Notify callbacks
        candle_data = {
            'timeframe': timeframe,
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume,
            'vwap': candle.vwap,
            'body_size': candle.body_size,
            'is_bullish': candle.is_bullish,
            'is_bearish': candle.is_bearish,
            'range': candle.range
        }
        
        for callback in self.candle_callbacks:
            try:
                callback(candle_data)
            except Exception as e:
                logging.error(f"Candle callback error: {e}")
        
        logging.debug(f"✅ {timeframe} candle completed: {candle.close:.2f} | Vol: {candle.volume:.2f}")
    
    def _update_1m_indicators(self, candle: BTCCandle):
        """Update technical indicators from 1m candles"""
        
        close_price = candle.close
        
        # Simple Moving Averages
        if len(self.candles_1m) >= 9:
            recent_9 = [c.close for c in list(self.candles_1m)[-9:]]
            self.sma_9_1m.append(np.mean(recent_9))
        
        if len(self.candles_1m) >= 20:
            recent_20 = [c.close for c in list(self.candles_1m)[-20:]]
            self.sma_20_1m.append(np.mean(recent_20))
        
        # Exponential Moving Averages
        if len(self.ema_12_1m) == 0:
            self.ema_12_1m.append(close_price)
        else:
            alpha = 2 / (12 + 1)
            ema_12 = alpha * close_price + (1 - alpha) * self.ema_12_1m[-1]
            self.ema_12_1m.append(ema_12)
        
        if len(self.ema_26_1m) == 0:
            self.ema_26_1m.append(close_price)
        else:
            alpha = 2 / (26 + 1)
            ema_26 = alpha * close_price + (1 - alpha) * self.ema_26_1m[-1]
            self.ema_26_1m.append(ema_26)
        
        # RSI (14 period)
        if len(self.candles_1m) >= 15:
            closes = [c.close for c in list(self.candles_1m)[-15:]]
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            self.rsi_14_1m.append(rsi)
    
    def get_swing_metrics(self) -> Dict:
        """Get comprehensive swing trading metrics"""
        
        if len(self.candles_1m) < 20:
            return {'insufficient_data': True}
        
        recent_1m = list(self.candles_1m)[-20:]
        recent_3m = list(self.candles_3m)[-10:] if len(self.candles_3m) >= 10 else []
        
        # Price metrics
        current_price = self.current_price
        
        # Moving average alignment
        ma_alignment = self._get_ma_alignment()
        
        # Market structure
        trend_direction = self.market_structure.trend_direction
        support_resistance = self._get_support_resistance_levels()
        
        # Volume analysis
        volume_analysis = self._analyze_volume_profile()
        
        # Multi-timeframe momentum
        momentum_1m = self._calculate_momentum(recent_1m, '1m')
        momentum_3m = self._calculate_momentum(recent_3m, '3m') if recent_3m else 0
        
        # RSI conditions
        current_rsi = self.rsi_14_1m[-1] if self.rsi_14_1m else 50
        
        # Volatility (average true range)
        atr = self._calculate_atr(recent_1m)
        
        return {
            'current_price': current_price,
            'trend_direction': trend_direction,
            'ma_alignment': ma_alignment,
            'momentum_1m': momentum_1m,
            'momentum_3m': momentum_3m,
            'current_rsi': current_rsi,
            'atr': atr,
            'support_levels': list(support_resistance['support']),
            'resistance_levels': list(support_resistance['resistance']),
            'volume_surge': volume_analysis['surge'],
            'vwap_position': volume_analysis['vwap_position'],
            'swing_highs': len(self.market_structure.swing_highs),
            'swing_lows': len(self.market_structure.swing_lows),
            'candles_1m_count': len(self.candles_1m),
            'candles_3m_count': len(self.candles_3m),
            'data_source': self.data_source,
            'structure_last_update': self.market_structure.last_structure_update
        }
    
    def _get_ma_alignment(self) -> Dict:
        """Get moving average alignment for trend confirmation"""
        
        if not self.sma_9_1m or not self.sma_20_1m or not self.ema_12_1m or not self.ema_26_1m:
            return {'aligned': False, 'direction': 'neutral'}
        
        current_price = self.current_price
        sma_9 = self.sma_9_1m[-1]
        sma_20 = self.sma_20_1m[-1]
        ema_12 = self.ema_12_1m[-1]
        ema_26 = self.ema_26_1m[-1]
        
        # Check for bullish alignment
        bullish_aligned = (current_price > sma_9 > sma_20 and 
                          ema_12 > ema_26)
        
        # Check for bearish alignment
        bearish_aligned = (current_price < sma_9 < sma_20 and 
                          ema_12 < ema_26)
        
        if bullish_aligned:
            return {'aligned': True, 'direction': 'bullish'}
        elif bearish_aligned:
            return {'aligned': True, 'direction': 'bearish'}
        else:
            return {'aligned': False, 'direction': 'mixed'}
    
    def _get_support_resistance_levels(self) -> Dict:
        """Get current support and resistance levels"""
        
        current_price = self.current_price
        
        # Get recent support/resistance from market structure
        support_levels = [level for level in self.market_structure.support_levels 
                         if level < current_price and current_price - level < 500]
        resistance_levels = [level for level in self.market_structure.resistance_levels 
                           if level > current_price and level - current_price < 500]
        
        # Sort by proximity to current price
        support_levels.sort(reverse=True)  # Closest support first
        resistance_levels.sort()  # Closest resistance first
        
        return {
            'support': support_levels[:3],  # Top 3 support levels
            'resistance': resistance_levels[:3]  # Top 3 resistance levels
        }
    
    def _analyze_volume_profile(self) -> Dict:
        """Analyze volume profile and VWAP position"""
        
        if len(self.candles_1m) < 10:
            return {'surge': False, 'vwap_position': 'neutral'}
        
        recent_candles = list(self.candles_1m)[-10:]
        current_price = self.current_price
        
        # Volume surge detection
        recent_volumes = [c.volume for c in recent_candles[-3:]]
        avg_volume = np.mean([c.volume for c in recent_candles[:-3]])
        current_volume = recent_volumes[-1] if recent_volumes else 0
        
        volume_surge = current_volume > avg_volume * 1.5
        
        # VWAP position
        recent_vwap = recent_candles[-1].vwap
        vwap_position = 'above' if current_price > recent_vwap else 'below'
        
        return {
            'surge': volume_surge,
            'vwap_position': vwap_position,
            'current_volume': current_volume,
            'avg_volume': avg_volume
        }
    
    def _calculate_momentum(self, candles: List[BTCCandle], timeframe: str) -> float:
        """Calculate momentum for given timeframe"""
        
        if len(candles) < 5:
            return 0.0
        
        # Price change over different periods
        if timeframe == '1m':
            short_period = 3
            long_period = 5
        else:  # 3m
            short_period = 2
            long_period = 3
        
        if len(candles) >= long_period:
            recent_close = candles[-1].close
            past_close = candles[-long_period].close
            momentum = (recent_close - past_close) / past_close
        else:
            momentum = 0.0
        
        return momentum
    
    def _calculate_atr(self, candles: List[BTCCandle], period: int = 14) -> float:
        """Calculate Average True Range for volatility measurement"""
        
        if len(candles) < period:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(candles)):
            current = candles[i]
            previous = candles[i-1]
            
            tr1 = current.high - current.low
            tr2 = abs(current.high - previous.close)
            tr3 = abs(current.low - previous.close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        # Return average of last 'period' true ranges
        recent_trs = true_ranges[-period:]
        return np.mean(recent_trs)
    
    def get_current_price(self) -> float:
        """Get current BTC price"""
        return self.current_price
    
    def get_connection_status(self) -> Dict:
        """Get data connection status"""
        return {
            'source': self.data_source,
            'tick_count': self.tick_count,
            'current_price': self.current_price,
            'connection_stable': self.connection_stable,
            'candles_1m': len(self.candles_1m),
            'candles_3m': len(self.candles_3m),
            'trend_direction': self.market_structure.trend_direction
        }
    
    def stop_data_feed(self):
        """Stop BTC data collection"""
        self.is_running = False
        self.connection_stable = False
        
        if hasattr(self, 'alpaca_api') and self.alpaca_api:
            try:
                self.alpaca_api.close()
            except:
                pass
        
        logging.info("📡 BTC swing data feed stopped")


if __name__ == "__main__":
    # Test BTC swing data collection
    collector = BTCSwingDataCollector()
    
    def on_candle_completed(candle_data):
        print(f"🕯️ {candle_data['timeframe']} candle: {candle_data['close']:.2f} "
              f"| Range: {candle_data['range']:.2f} | Vol: {candle_data['volume']:.2f}")
    
    collector.add_candle_callback(on_candle_completed)
    collector.start_data_feed()
    
    try:
        print("Testing BTC swing data collection for 60 seconds...")
        time.sleep(60)
        
        # Check swing metrics
        metrics = collector.get_swing_metrics()
        print(f"\n📊 Swing Metrics:")
        for key, value in metrics.items():
            if not isinstance(value, (list, dict)):
                print(f"   {key}: {value}")
        
        print("✅ BTC swing data collection test completed!")
        
    except KeyboardInterrupt:
        print("\n🛑 Test stopped")
    finally:
        collector.stop_data_feed()