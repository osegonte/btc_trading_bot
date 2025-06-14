#!/usr/bin/env python3
"""
BTC Tick Data Collection - Core File 1/4
Purpose: Connect to Alpaca API and stream live BTC tick data
Ensures constant real-time data feed for scalping decisions
"""

import time
import logging
import threading
import numpy as np
import json
from datetime import datetime
from collections import deque
from typing import Dict, Optional, Callable


class BTCTickData:
    """BTC tick data structure for scalping"""
    def __init__(self, price: float, size: float, timestamp: datetime, source: str = "alpaca"):
        self.price = price
        self.size = size
        self.timestamp = timestamp
        self.source = source
        self.bid = price - 0.50
        self.ask = price + 0.50
        self.spread = self.ask - self.bid


class BTCDataCollector:
    """
    Core BTC data collector for scalping bot
    Streams live tick data from Alpaca with simulation fallback
    """
    
    def __init__(self, symbol: str = "BTCUSD"):
        self.symbol = symbol
        self.is_running = False
        
        # Tick management for scalping
        self.tick_buffer = deque(maxlen=100)  # Keep last 100 ticks for analysis
        self.tick_callbacks = []
        self.current_price = 43000.0
        self.tick_count = 0
        self.data_source = "none"
        
        # Connection management
        self.alpaca_api = None
        self.connection_stable = False
        self.simulation_active = False
        
        # Scalping-specific analytics
        self.price_history = deque(maxlen=50)  # Short history for fast decisions
        self.volume_history = deque(maxlen=50)
        self.last_tick_time = None
        self.ticks_per_second = 0
        
        logging.info(f"✅ BTC Data Collector initialized for {symbol}")
    
    def add_tick_callback(self, callback: Callable):
        """Add callback function for tick updates"""
        self.tick_callbacks.append(callback)
        logging.debug(f"Added tick callback, total: {len(self.tick_callbacks)}")
    
    def start_data_feed(self, api_key: str = "", secret_key: str = ""):
        """
        Start BTC data feed with Alpaca primary, simulation fallback
        Ensures data flow for scalping strategy
        """
        self.is_running = True
        logging.info("🚀 Starting BTC tick data feed...")
        
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
        
        # Fallback to guaranteed simulation
        logging.info("🎮 Starting BTC simulation for scalping...")
        self._start_simulation_feed()
    
    def _start_alpaca_feed(self, api_key: str, secret_key: str):
        """Start Alpaca crypto data feed"""
        try:
            import alpaca_trade_api as tradeapi
            
            # Use paper trading URL for crypto
            self.alpaca_api = tradeapi.StreamConn(
                api_key, 
                secret_key,
                base_url='https://paper-api.alpaca.markets',
                data_feed='crypto'
            )
            
            @self.alpaca_api.on(f'^T\\.{self.symbol}')
            async def on_btc_tick(conn, channel, data):
                """Handle incoming BTC tick from Alpaca"""
                tick = BTCTickData(
                    price=float(data.price),
                    size=float(data.size),
                    timestamp=datetime.now(),
                    source="alpaca"
                )
                self._process_tick(tick)
            
            def run_alpaca_stream():
                """Run Alpaca stream in thread"""
                try:
                    self.alpaca_api.run()
                except Exception as e:
                    logging.error(f"Alpaca stream error: {e}")
                    # Switch to simulation on error
                    if self.is_running:
                        self._start_simulation_feed()
            
            # Start Alpaca stream thread
            alpaca_thread = threading.Thread(target=run_alpaca_stream, daemon=True)
            alpaca_thread.start()
            self.data_source = "alpaca"
            self.connection_stable = True
            
        except ImportError:
            raise Exception("alpaca-trade-api not installed")
        except Exception as e:
            raise Exception(f"Alpaca setup failed: {e}")
    
    def _start_simulation_feed(self):
        """Start simulation feed optimized for BTC scalping"""
        
        def simulate_btc_scalping_ticks():
            """Generate realistic BTC ticks for scalping"""
            base_price = 43000.0
            momentum = 0.0
            trend_duration = 0
            volatility_state = "normal"  # normal, high, low
            
            logging.info(f"🎮 BTC scalping simulation started at ${base_price:,.2f}")
            
            while self.is_running:
                try:
                    # Trend management for scalping opportunities
                    if trend_duration <= 0:
                        trend_duration = np.random.randint(15, 45)  # 15-45 ticks
                        momentum = np.random.uniform(-0.8, 0.8)
                        
                        # Volatility state changes
                        vol_rand = np.random.random()
                        if vol_rand < 0.1:
                            volatility_state = "high"
                        elif vol_rand < 0.2:
                            volatility_state = "low"
                        else:
                            volatility_state = "normal"
                    
                    # Price movement calculation
                    base_volatility = {
                        "low": 5.0,
                        "normal": 15.0,
                        "high": 35.0
                    }[volatility_state]
                    
                    # Momentum component
                    momentum_move = momentum * 2.0
                    
                    # Random component
                    random_move = np.random.normal(0, base_volatility / 3)
                    
                    # Occasional spikes (scalping opportunities)
                    if np.random.random() < 0.05:  # 5% chance
                        spike_direction = 1 if np.random.random() > 0.5 else -1
                        spike_magnitude = np.random.uniform(20, 50)
                        random_move += spike_direction * spike_magnitude
                    
                    # Apply total movement
                    total_move = momentum_move + random_move
                    base_price += total_move
                    
                    # Keep in realistic range
                    base_price = max(35000, min(55000, base_price))
                    
                    # Generate volume (higher during spikes)
                    if abs(total_move) > 20:
                        volume = np.random.uniform(1.0, 5.0)  # High volume on big moves
                    else:
                        volume = np.random.uniform(0.1, 2.0)  # Normal volume
                    
                    # Create and process tick
                    tick = BTCTickData(
                        price=round(base_price, 2),
                        size=round(volume, 4),
                        timestamp=datetime.now(),
                        source="simulation"
                    )
                    
                    self._process_tick(tick)
                    
                    trend_duration -= 1
                    
                    # Fast tick rate for scalping (10 ticks/second)
                    time.sleep(0.1)
                    
                except Exception as e:
                    logging.error(f"Simulation error: {e}")
                    time.sleep(1)
        
        # Start simulation thread
        sim_thread = threading.Thread(target=simulate_btc_scalping_ticks, daemon=True)
        sim_thread.start()
        self.data_source = "simulation"
        self.connection_stable = True
        self.simulation_active = True
    
    def _process_tick(self, tick: BTCTickData):
        """Process incoming BTC tick for scalping"""
        try:
            # Store tick
            self.tick_buffer.append(tick)
            self.current_price = tick.price
            self.tick_count += 1
            
            # Update scalping analytics
            self.price_history.append(tick.price)
            self.volume_history.append(tick.size)
            
            # Calculate tick rate
            current_time = time.time()
            if self.last_tick_time:
                time_diff = current_time - self.last_tick_time
                if time_diff > 0:
                    self.ticks_per_second = 0.9 * self.ticks_per_second + 0.1 * (1.0 / time_diff)
            self.last_tick_time = current_time
            
            # Create tick data for trading logic
            tick_data = {
                'price': tick.price,
                'size': tick.size,
                'timestamp': tick.timestamp,
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.spread,
                'symbol': self.symbol,
                'data_source': tick.source,
                'tick_count': self.tick_count,
                'ticks_per_second': self.ticks_per_second
            }
            
            # Send to all callbacks (trading logic)
            for callback in self.tick_callbacks:
                try:
                    callback(tick_data)
                except Exception as e:
                    logging.error(f"Callback error: {e}")
            
        except Exception as e:
            logging.error(f"Tick processing error: {e}")
    
    def get_current_price(self) -> float:
        """Get current BTC price for scalping decisions"""
        return self.current_price
    
    def get_scalping_metrics(self) -> Dict:
        """Get real-time metrics for scalping strategy"""
        if len(self.tick_buffer) < 10:
            return {'insufficient_data': True}
        
        prices = np.array([tick.price for tick in self.tick_buffer])
        volumes = np.array([tick.size for tick in self.tick_buffer])
        
        # Short-term scalping indicators
        price_change_1min = self.get_price_change(10) if len(prices) >= 10 else 0  # 10 ticks ≈ 1 minute
        price_change_30sec = self.get_price_change(5) if len(prices) >= 5 else 0   # 5 ticks ≈ 30 seconds
        
        # Momentum for scalping
        momentum_fast = (prices[-1] - prices[-3]) / prices[-3] if len(prices) >= 3 else 0
        momentum_medium = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        
        # Volume spike detection
        recent_volume = np.mean(volumes[-3:]) if len(volumes) >= 3 else volumes[-1]
        avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else recent_volume
        volume_spike = recent_volume > avg_volume * 1.5 if avg_volume > 0 else False
        
        # Volatility (important for scalping)
        price_volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 else 0
        
        return {
            'current_price': prices[-1],
            'price_change_1min': price_change_1min,
            'price_change_30sec': price_change_30sec,
            'momentum_fast': momentum_fast,
            'momentum_medium': momentum_medium,
            'volume_spike': volume_spike,
            'price_volatility': price_volatility,
            'tick_count': len(self.tick_buffer),
            'data_source': self.data_source,
            'connection_stable': self.connection_stable,
            'ticks_per_second': self.ticks_per_second,
            'recent_volume': recent_volume,
            'avg_volume': avg_volume
        }
    
    def get_price_change(self, periods: int) -> float:
        """Get price change over specified periods"""
        if len(self.tick_buffer) < periods:
            return 0.0
        
        current = self.tick_buffer[-1].price
        past = self.tick_buffer[-periods].price
        
        return ((current - past) / past) * 100 if past != 0 else 0.0
    
    def get_connection_status(self) -> Dict:
        """Get data connection status for monitoring"""
        return {
            'source': self.data_source,
            'tick_count': self.tick_count,
            'current_price': self.current_price,
            'connection_stable': self.connection_stable,
            'simulation_active': self.simulation_active,
            'ticks_per_second': round(self.ticks_per_second, 2),
            'buffer_size': len(self.tick_buffer)
        }
    
    def stop_data_feed(self):
        """Stop BTC data collection"""
        self.is_running = False
        self.connection_stable = False
        
        if self.alpaca_api:
            try:
                self.alpaca_api.close()
            except:
                pass
        
        logging.info("📡 BTC data feed stopped")


if __name__ == "__main__":
    # Test BTC data collection
    collector = BTCDataCollector()
    
    def on_tick(tick_data):
        print(f"₿ {tick_data['price']:,.2f} | Vol: {tick_data['size']:.2f} | Source: {tick_data['data_source']}")
    
    collector.add_tick_callback(on_tick)
    collector.start_data_feed()
    
    try:
        print("Testing BTC data collection for 30 seconds...")
        time.sleep(30)
        
        # Check results
        status = collector.get_connection_status()
        metrics = collector.get_scalping_metrics()
        
        print(f"\n📊 Status: {status}")
        print(f"📈 Metrics: {metrics}")
        
        if status['tick_count'] > 0:
            print("✅ BTC data collection working!")
        else:
            print("❌ No data received!")
            
    except KeyboardInterrupt:
        print("\n🛑 Test stopped")
    finally:
        collector.stop_data_feed()