#!/usr/bin/env python3
"""
BTC Bot Startup Script
"""

import sys
import os

def main():
    print("₿ Starting Bitcoin Trading Bot...")
    print("=" * 50)
    
    # Check if we're in the right directory
    required_files = ['config.py', 'main.py', 'data_collection.py', 'trading_logic.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all bot files are in the current directory.")
        return
    
    # Import and run the main bot
    try:
        import asyncio
        from main import main as bot_main
        
        print("✅ All required files found")
        print("🚀 Launching BTC trading bot...")
        print("⏹️  Press Ctrl+C to stop\n")
        
        # Run the bot
        asyncio.run(bot_main())
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install dependencies with: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error starting bot: {e}")

if __name__ == "__main__":
    main()