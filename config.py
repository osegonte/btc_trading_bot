#!/usr/bin/env python3
"""
Simple Configuration for BTC/USD Tick Trading Bot
"""

# Alpaca API Configuration
ALPACA_CONFIG = {
    'paper_trading': True,
    'api_key': 'PK43BTKX4DJCXAVB5BIS',  # Replace with your keys
    'secret_key': 'LOulDBLtPY9H3z6TfXMMCzDPTtjBXjI59pxD2So5',  # Replace with your keys
    'symbol': 'BTCUSD',
    'quantity': 0.001  # 0.001 BTC = $50-100 depending on price
}

# Alternative data sources for weekend testing
DATA_CONFIG = {
    'use_alpaca': True,          # Set to False for weekend testing
    'use_coinbase': False,       # Enable for real crypto data
    'use_simulation': True,      # Fallback simulation mode
    'websocket_feeds': {
        'coinbase': 'wss://ws-feed.pro.coinbase.com',
        'binance': 'wss://stream.binance.com:9443/ws/btcusdt@ticker'
    }
}

# Trading Strategy Settings (adjusted for BTC volatility)
STRATEGY_CONFIG = {
    'profit_target_ticks': 8,    # 8 ticks = $8.00 profit target (BTC more volatile)
    'stop_loss_ticks': 4,        # 4 ticks = $4.00 stop loss
    'tick_size': 1.0,            # BTC/USD tick size = $1.00
    'min_confidence': 0.70       # Slightly higher confidence for crypto
}

# Bot Settings
BOT_CONFIG = {
    'log_file': 'btcusd_trades.csv',
    'max_position_time': 30,     # Max 30 seconds in position (crypto moves fast)
    'status_update_interval': 5, # Status updates every 5 seconds
    'max_daily_trades': 50,      # Higher limit for crypto
    'weekend_mode': True         # Enable special weekend testing features
}

# Weekend Testing Configuration
WEEKEND_CONFIG = {
    'enable_simulation': True,
    'base_price': 43000.0,      # Starting BTC price for simulation
    'volatility_multiplier': 2.0, # BTC is more volatile than gold
    'tick_frequency': 0.1,      # Faster ticks (10 per second)
    'price_variation': 50.0,    # ±$50 price swings
    'volume_range': (1, 100)    # Volume range for simulation
}