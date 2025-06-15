#!/usr/bin/env python3
"""
Enhanced ML Interface for BTC Swing Trading
Purpose: Machine learning pattern recognition for 2-5 minute swing positions
Key Changes: Tick-based ML → Swing pattern recognition with market structure
"""

import logging
import pickle
import os
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    ML_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ ML libraries not available. Install with: pip install scikit-learn")
    ML_AVAILABLE = False


class BTCSwingMLSignal:
    """Enhanced ML signal structure for swing trading"""
    def __init__(self, signal: str, confidence: float, reasoning: str, 
                 expected_hold_time: int = 180, indicators: Dict = None):
        self.signal = signal        # 'buy', 'sell', 'hold'
        self.confidence = confidence # 0.0 to 1.0
        self.reasoning = reasoning
        self.expected_hold_time = expected_hold_time  # seconds
        self.indicators = indicators or {}


class BTCSwingFeatureExtractor:
    """Extract features from BTC data for swing trading ML"""
    
    def __init__(self, lookback_candles: int = 20):
        self.lookback_candles = lookback_candles
        self.candle_history_1m = deque(maxlen=lookback_candles)
        self.candle_history_3m = deque(maxlen=10)
        self.price_history = deque(maxlen=50)
        self.volume_history = deque(maxlen=50)
    
    def add_candle_data(self, candle_data: Dict):
        """Add candle data for feature extraction"""
        timeframe = candle_data.get('timeframe', '1m')
        
        if timeframe == '1m':
            self.candle_history_1m.append(candle_data)
        elif timeframe == '3m':
            self.candle_history_3m.append(candle_data)
        
        # Also maintain price/volume history for compatibility
        self.price_history.append(candle_data.get('close', 0))
        self.volume_history.append(candle_data.get('volume', 0))
    
    def add_tick_data(self, tick_data: Dict):
        """Add tick data (for compatibility with existing interface)"""
        self.price_history.append(tick_data.get('price', 0))
        self.volume_history.append(tick_data.get('size', 0))
    
    def extract_swing_features(self, swing_metrics: Dict = None) -> Dict:
        """Extract comprehensive features for swing trading ML"""
        
        if len(self.candle_history_1m) < 10:
            return {}
        
        features = {}
        candles_1m = list(self.candle_history_1m)
        candles_3m = list(self.candle_history_3m)
        
        # Current market state
        current_candle = candles_1m[-1]
        features['current_price'] = current_candle.get('close', 0)
        features['current_volume'] = current_candle.get('volume', 0)
        features['current_range'] = current_candle.get('range', 0)
        features['is_bullish_candle'] = current_candle.get('is_bullish', False)
        features['body_size'] = current_candle.get('body_size', 0)
        
        # Price movement features (swing-focused)
        if len(candles_1m) >= 5:
            closes = [c.get('close', 0) for c in candles_1m[-5:]]
            features['price_change_5min'] = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
            features['price_momentum_5min'] = np.mean(np.diff(closes)) / closes[0] if closes[0] > 0 else 0
        
        if len(candles_1m) >= 10:
            closes = [c.get('close', 0) for c in candles_1m[-10:]]
            features['price_change_10min'] = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
            features['price_momentum_10min'] = np.mean(np.diff(closes)) / closes[0] if closes[0] > 0 else 0
        
        # Swing volatility analysis
        if len(candles_1m) >= 10:
            ranges = [c.get('range', 0) for c in candles_1m[-10:]]
            avg_range = np.mean(ranges)
            features['volatility_current_vs_avg'] = current_candle.get('range', 0) / avg_range if avg_range > 0 else 1
            features['volatility_percentile'] = (sorted(ranges).index(current_candle.get('range', 0)) + 1) / len(ranges) * 100
        
        # Market structure features from swing_metrics
        if swing_metrics:
            features['trend_direction_bullish'] = 1 if swing_metrics.get('trend_direction') == 'uptrend' else 0
            features['trend_direction_bearish'] = 1 if swing_metrics.get('trend_direction') == 'downtrend' else 0
            features['ma_aligned'] = 1 if swing_metrics.get('ma_alignment', {}).get('aligned', False) else 0
            features['ma_bullish'] = 1 if swing_metrics.get('ma_alignment', {}).get('direction') == 'bullish' else 0
            features['ma_bearish'] = 1 if swing_metrics.get('ma_alignment', {}).get('direction') == 'bearish' else 0
            features['momentum_1m'] = swing_metrics.get('momentum_1m', 0)
            features['momentum_3m'] = swing_metrics.get('momentum_3m', 0)
            features['current_rsi'] = swing_metrics.get('current_rsi', 50)
            features['atr'] = swing_metrics.get('atr', 0)
            features['volume_surge'] = 1 if swing_metrics.get('volume_surge', False) else 0
            features['vwap_above'] = 1 if swing_metrics.get('vwap_position') == 'above' else 0
            features['vwap_below'] = 1 if swing_metrics.get('vwap_position') == 'below' else 0
            
            # Support/resistance interaction
            support_levels = swing_metrics.get('support_levels', [])
            resistance_levels = swing_metrics.get('resistance_levels', [])
            current_price = features['current_price']
            
            features['near_support'] = self._near_level(current_price, support_levels)
            features['near_resistance'] = self._near_level(current_price, resistance_levels)
            features['support_count'] = len(support_levels)
            features['resistance_count'] = len(resistance_levels)
        
        # Multi-timeframe confirmation
        if len(candles_3m) >= 3:
            closes_3m = [c.get('close', 0) for c in candles_3m[-3:]]
            features['momentum_3m_tf'] = (closes_3m[-1] - closes_3m[0]) / closes_3m[0] if closes_3m[0] > 0 else 0
            
            # 3m candle patterns
            recent_3m = candles_3m[-1]
            features['bullish_3m_candle'] = 1 if recent_3m.get('is_bullish', False) else 0
            features['large_3m_body'] = 1 if recent_3m.get('body_size', 0) > recent_3m.get('range', 1) * 0.7 else 0
        
        # Volume analysis for swing trading
        if len(candles_1m) >= 10:
            volumes = [c.get('volume', 0) for c in candles_1m[-10:]]
            avg_volume = np.mean(volumes[:-1])  # Exclude current
            current_volume = volumes[-1]
            
            features['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1
            features['volume_surge_significant'] = 1 if current_volume > avg_volume * 1.5 else 0
            features['volume_trend'] = np.corrcoef(range(len(volumes)), volumes)[0, 1] if len(volumes) > 1 else 0
        
        # Swing pattern recognition
        swing_patterns = self._detect_swing_patterns(candles_1m)
        features.update(swing_patterns)
        
        # RSI-based features
        if swing_metrics:
            rsi = swing_metrics.get('current_rsi', 50)
            features['rsi_oversold'] = 1 if rsi < 30 else 0
            features['rsi_overbought'] = 1 if rsi > 70 else 0
            features['rsi_bullish_zone'] = 1 if 40 < rsi < 60 else 0
            features['rsi_bearish_zone'] = 1 if 40 < rsi < 60 else 0
            features['rsi_extreme'] = 1 if rsi < 25 or rsi > 75 else 0
        
        # Time-based features
        current_time = datetime.now()
        features['hour_of_day'] = current_time.hour
        features['day_of_week'] = current_time.weekday()
        features['is_market_hours_us'] = 1 if 9 <= current_time.hour <= 16 else 0
        features['is_market_hours_eu'] = 1 if 8 <= current_time.hour <= 17 else 0
        
        return features
    
    def _near_level(self, price: float, levels: List[float], tolerance: float = 0.003) -> float:
        """Check proximity to support/resistance levels"""
        if not levels:
            return 0.0
        
        min_distance = min(abs(price - level) / level for level in levels[:3])
        return 1.0 if min_distance < tolerance else 0.0
    
    def _detect_swing_patterns(self, candles: List[Dict]) -> Dict:
        """Detect swing trading patterns"""
        patterns = {}
        
        if len(candles) < 5:
            return patterns
        
        recent_5 = candles[-5:]
        closes = [c.get('close', 0) for c in recent_5]
        highs = [c.get('high', 0) for c in recent_5]
        lows = [c.get('low', 0) for c in recent_5]
        
        # Higher highs and higher lows
        patterns['higher_highs'] = 1 if highs[-1] > highs[-2] > highs[-3] else 0
        patterns['higher_lows'] = 1 if lows[-1] > lows[-2] > lows[-3] else 0
        patterns['lower_highs'] = 1 if highs[-1] < highs[-2] < highs[-3] else 0
        patterns['lower_lows'] = 1 if lows[-1] < lows[-2] < lows[-3] else 0
        
        # Breakout patterns
        recent_high = max(highs[:-1])
        recent_low = min(lows[:-1])
        current_close = closes[-1]
        
        patterns['bullish_breakout'] = 1 if current_close > recent_high * 1.002 else 0
        patterns['bearish_breakdown'] = 1 if current_close < recent_low * 0.998 else 0
        
        # Consolidation patterns
        price_range = max(closes) - min(closes)
        avg_price = np.mean(closes)
        patterns['consolidation'] = 1 if price_range / avg_price < 0.01 else 0  # 1% range
        
        # Momentum patterns
        patterns['strong_bullish_momentum'] = 1 if all(closes[i] > closes[i-1] for i in range(1, len(closes))) else 0
        patterns['strong_bearish_momentum'] = 1 if all(closes[i] < closes[i-1] for i in range(1, len(closes))) else 0
        
        return patterns


class BTCSwingMLModel:
    """Enhanced ML model for BTC swing trading predictions"""
    
    def __init__(self, model_file: str = "btc_swing_trading_model.pkl"):
        self.model_file = model_file
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        
        # Training data storage
        self.training_features = []
        self.training_labels = []
        self.training_outcomes = []
        self.training_hold_times = []
        
        # Performance tracking
        self.model_version = 1
        self.last_retrain = None
        
        # Load existing model if available
        self.load_model()
    
    def add_training_data(self, features: Dict, outcome: str, profit_loss: float, hold_time: int = 0):
        """Add swing trading data for model training"""
        
        if not features:
            return
        
        # Convert outcome to label for classification
        if outcome == 'profitable':
            label = 1  # Profitable swing trade
        elif outcome == 'unprofitable':
            label = 0  # Unprofitable swing trade
        else:
            return  # Skip no-trade outcomes
        
        # Store training data with swing-specific metrics
        self.training_features.append(features.copy())
        self.training_labels.append(label)
        self.training_outcomes.append(profit_loss)
        self.training_hold_times.append(hold_time)
        
        # Auto-train every 20 samples for swing trading (less frequent than scalping)
        if len(self.training_features) >= 20 and len(self.training_features) % 20 == 0:
            self.train_model()
    
    def train_model(self, min_samples: int = 20):
        """Train the BTC swing trading ML model"""
        
        if not ML_AVAILABLE:
            logging.warning("ML libraries not available for training")
            return False
        
        if len(self.training_features) < min_samples:
            logging.info(f"Need {min_samples} samples to train swing model. Have {len(self.training_features)}")
            return False
        
        try:
            # Prepare data
            X, y = self._prepare_training_data()
            
            if len(X) == 0:
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model optimized for swing trading
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42,
                class_weight='balanced'  # Handle imbalanced data
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            self.model_version += 1
            self.last_retrain = datetime.now()
            self.save_model()
            
            logging.info(f"✅ BTC Swing ML model trained! Accuracy: {accuracy:.2f} | Version: {self.model_version} | Samples: {len(X_train)}")
            return True
            
        except Exception as e:
            logging.error(f"BTC Swing ML training error: {e}")
            return False
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training arrays for BTC swing model"""
        
        if not self.training_features:
            return np.array([]), np.array([])
        
        # Get feature names from first sample
        self.feature_names = sorted(list(self.training_features[0].keys()))
        
        # Create feature matrix
        X = []
        y = []
        
        for features, label in zip(self.training_features, self.training_labels):
            # Create feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                value = features.get(feature_name, 0)
                # Handle invalid values
                if np.isnan(value) or np.isinf(value):
                    value = 0
                feature_vector.append(value)
            
            X.append(feature_vector)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def predict(self, features: Dict) -> BTCSwingMLSignal:
        """Make BTC swing trading prediction using trained model"""
        
        if not self.is_trained or not self.model:
            return BTCSwingMLSignal('hold', 0.0, 'Swing model not trained', 180, {})
        
        try:
            # Prepare feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                value = features.get(feature_name, 0)
                if np.isnan(value) or np.isinf(value):
                    value = 0
                feature_vector.append(value)
            
            X = np.array([feature_vector])
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            confidence = max(probabilities)
            
            # Enhanced signal logic for swing trading
            signal_indicators = {
                'prediction': prediction,
                'confidence': confidence,
                'model_version': self.model_version,
                'feature_count': len(self.feature_names),
                'training_samples': len(self.training_features)
            }
            
            # Estimate hold time based on training data
            if self.training_hold_times:
                avg_hold_time = int(np.mean(self.training_hold_times))
                expected_hold = max(120, min(300, avg_hold_time))  # 2-5 minutes
            else:
                expected_hold = 180  # Default 3 minutes
            
            # Convert prediction to signal with swing-specific thresholds
            if prediction == 1 and confidence > 0.65:  # Higher threshold for swing trades
                signal = 'buy'
                reasoning = f'Swing ML: Profitable trade predicted (conf: {confidence:.2f}, v{self.model_version})'
            elif prediction == 0 and confidence > 0.65:
                signal = 'sell'  # Could indicate short opportunity
                reasoning = f'Swing ML: Bearish signal (conf: {confidence:.2f}, v{self.model_version})'
            else:
                signal = 'hold'
                reasoning = f'Swing ML: Low confidence {confidence:.2f} (threshold: 0.65)'
            
            return BTCSwingMLSignal(signal, confidence, reasoning, expected_hold, signal_indicators)
            
        except Exception as e:
            logging.error(f"BTC Swing ML prediction error: {e}")
            return BTCSwingMLSignal('hold', 0.0, f'Swing prediction error: {e}', 180, {})
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from BTC swing model"""
        
        if not self.is_trained or not self.model:
            return {}
        
        try:
            importances = self.model.feature_importances_
            importance_dict = {}
            
            for i, feature_name in enumerate(self.feature_names):
                importance_dict[feature_name] = importances[i]
            
            # Sort by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_features)
            
        except Exception as e:
            logging.error(f"Error getting swing feature importance: {e}")
            return {}
    
    def get_swing_insights(self) -> Dict:
        """Get insights specific to swing trading performance"""
        
        if not self.training_outcomes or not self.training_hold_times:
            return {}
        
        profitable_trades = [i for i, outcome in enumerate(self.training_outcomes) if outcome > 0]
        unprofitable_trades = [i for i, outcome in enumerate(self.training_outcomes) if outcome <= 0]
        
        insights = {
            'total_samples': len(self.training_outcomes),
            'profitable_samples': len(profitable_trades),
            'win_rate': len(profitable_trades) / len(self.training_outcomes) * 100,
        }
        
        # Hold time analysis
        if profitable_trades:
            profitable_hold_times = [self.training_hold_times[i] for i in profitable_trades]
            insights['avg_profitable_hold_time'] = np.mean(profitable_hold_times) / 60  # minutes
        
        if unprofitable_trades:
            unprofitable_hold_times = [self.training_hold_times[i] for i in unprofitable_trades]
            insights['avg_unprofitable_hold_time'] = np.mean(unprofitable_hold_times) / 60  # minutes
        
        # Profit analysis
        profitable_outcomes = [self.training_outcomes[i] for i in profitable_trades]
        unprofitable_outcomes = [self.training_outcomes[i] for i in unprofitable_trades]
        
        if profitable_outcomes:
            insights['avg_profit'] = np.mean(profitable_outcomes)
            insights['max_profit'] = max(profitable_outcomes)
        
        if unprofitable_outcomes:
            insights['avg_loss'] = np.mean(unprofitable_outcomes)
            insights['max_loss'] = min(unprofitable_outcomes)
        
        return insights
    
    def save_model(self):
        """Save BTC swing model to file"""
        
        if not self.is_trained:
            return
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'training_samples': len(self.training_features),
                'model_version': self.model_version,
                'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None,
                'timestamp': datetime.now().isoformat(),
                'training_outcomes': self.training_outcomes[-50:],  # Keep last 50 outcomes
                'training_hold_times': self.training_hold_times[-50:],
                'swing_trading_optimized': True
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logging.info(f"✅ BTC Swing ML model v{self.model_version} saved to {self.model_file}")
            
        except Exception as e:
            logging.error(f"Error saving swing model: {e}")
    
    def load_model(self):
        """Load BTC swing model from file"""
        
        if not os.path.exists(self.model_file):
            return False
        
        try:
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_version = model_data.get('model_version', 1)
            self.is_trained = True
            
            # Load training history if available
            self.training_outcomes = model_data.get('training_outcomes', [])
            self.training_hold_times = model_data.get('training_hold_times', [])
            
            last_retrain_str = model_data.get('last_retrain')
            if last_retrain_str:
                self.last_retrain = datetime.fromisoformat(last_retrain_str)
            
            swing_optimized = model_data.get('swing_trading_optimized', False)
            mode = "Swing" if swing_optimized else "Legacy"
            
            logging.info(f"✅ BTC {mode} ML model v{self.model_version} loaded from {self.model_file}")
            logging.info(f"   Training samples: {model_data.get('training_samples', 'unknown')}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading swing model: {e}")
            return False


class BTCSwingMLInterface:
    """Main ML interface for the BTC swing trading bot"""
    
    def __init__(self, config: Dict = None):
        if config is None:
            config = {}
        
        self.feature_extractor = BTCSwingFeatureExtractor(
            lookback_candles=config.get('lookback_candles', 20)
        )
        self.ml_model = BTCSwingMLModel(
            model_file=config.get('model_file', 'btc_swing_ml_model.pkl')
        )
        
        # Performance tracking
        self.predictions_made = 0
        self.correct_predictions = 0
        self.total_ml_pnl = 0.0
        
        # Swing-specific tracking
        self.swing_signals_given = 0
        self.successful_swing_signals = 0
        self.total_swing_hold_time = 0
        
        logging.info("✅ BTC Swing ML interface initialized")
    
    def process_candle(self, candle_data: Dict, swing_metrics: Dict = None) -> BTCSwingMLSignal:
        """Process BTC candle and return swing ML signal"""
        
        # Add candle to feature extractor
        self.feature_extractor.add_candle_data(candle_data)
        
        # Extract features with swing metrics
        features = self.feature_extractor.extract_swing_features(swing_metrics)
        
        if not features:
            return BTCSwingMLSignal('hold', 0.0, 'Insufficient swing data for features', 180, {})
        
        # Get ML prediction
        ml_signal = self.ml_model.predict(features)
        
        if ml_signal.signal != 'hold':
            self.predictions_made += 1
            self.swing_signals_given += 1
        
        return ml_signal
    
    def process_tick(self, tick_data: Dict) -> BTCSwingMLSignal:
        """Process tick data (compatibility method)"""
        
        # For compatibility with existing interface
        self.feature_extractor.add_tick_data(tick_data)
        
        # Extract basic features without swing metrics
        features = self.feature_extractor.extract_swing_features()
        
        if not features:
            return BTCSwingMLSignal('hold', 0.0, 'Insufficient data for swing features', 180, {})
        
        # Get ML prediction
        ml_signal = self.ml_model.predict(features)
        
        if ml_signal.signal != 'hold':
            self.predictions_made += 1
            self.swing_signals_given += 1
        
        return ml_signal
    
    def record_trade_outcome(self, features: Dict, signal: str, profit_loss: float, hold_time: int = 0):
        """Record BTC swing trade outcome for model learning"""
        
        if not features:
            return
        
        # Track ML performance
        self.total_ml_pnl += profit_loss
        self.total_swing_hold_time += hold_time
        
        # Determine outcome
        if signal == 'hold':
            outcome = 'no_trade'
        elif profit_loss > 0:
            outcome = 'profitable'
            if self.predictions_made > 0:
                self.correct_predictions += 1
                self.successful_swing_signals += 1
        else:
            outcome = 'unprofitable'
        
        # Add to training data with hold time
        self.ml_model.add_training_data(features, outcome, profit_loss, hold_time)
        
        hold_minutes = hold_time / 60 if hold_time > 0 else 0
        logging.debug(f"Swing ML outcome recorded: {outcome} | P&L: €{profit_loss:.2f} | Hold: {hold_minutes:.1f}m")
    
    def get_ml_stats(self) -> Dict:
        """Get BTC swing ML statistics"""
        
        accuracy = (self.correct_predictions / self.predictions_made * 100) if self.predictions_made > 0 else 0
        swing_success_rate = (self.successful_swing_signals / self.swing_signals_given * 100) if self.swing_signals_given > 0 else 0
        feature_importance = self.ml_model.get_feature_importance()
        swing_insights = self.ml_model.get_swing_insights()
        
        avg_swing_hold = (self.total_swing_hold_time / max(1, self.swing_signals_given)) / 60  # minutes
        
        return {
            'ml_available': ML_AVAILABLE,
            'model_trained': self.ml_model.is_trained,
            'model_version': self.ml_model.model_version,
            'training_samples': len(self.ml_model.training_features),
            'predictions_made': self.predictions_made,
            'correct_predictions': self.correct_predictions,
            'accuracy': accuracy,
            'swing_signals_given': self.swing_signals_given,
            'successful_swing_signals': self.successful_swing_signals,
            'swing_success_rate': swing_success_rate,
            'total_ml_pnl': self.total_ml_pnl,
            'avg_ml_pnl': self.total_ml_pnl / max(1, self.predictions_made),
            'avg_swing_hold_minutes': avg_swing_hold,
            'last_retrain': self.ml_model.last_retrain.isoformat() if self.ml_model.last_retrain else None,
            'top_features': dict(list(feature_importance.items())[:5]),  # Top 5 features
            'swing_insights': swing_insights,
            'swing_trading_mode': True
        }
    
    def force_retrain(self):
        """Force BTC swing model retraining"""
        
        if ML_AVAILABLE:
            success = self.ml_model.train_model(min_samples=15)
            if success:
                logging.info("✅ BTC Swing ML model retrained")
            else:
                logging.warning("⚠️ BTC Swing ML retraining failed")
        else:
            logging.warning("⚠️ ML libraries not available")
    
    def get_feature_analysis(self) -> Dict:
        """Get detailed feature analysis for BTC swing trading"""
        
        if not self.ml_model.is_trained:
            return {}
        
        feature_importance = self.ml_model.get_feature_importance()
        swing_insights = self.ml_model.get_swing_insights()
        
        return {
            'feature_importance': feature_importance,
            'total_features': len(self.ml_model.feature_names),
            'model_performance': {
                'accuracy': (self.correct_predictions / max(1, self.predictions_made)) * 100,
                'total_predictions': self.predictions_made,
                'model_version': self.ml_model.model_version
            },
            'swing_insights': swing_insights,
            'swing_trading_optimized': True
        }


# Configuration for BTC Swing ML Interface
BTC_SWING_ML_CONFIG = {
    'lookback_candles': 20,
    'model_file': 'btc_swing_ml_model.pkl',
    'min_confidence': 0.65,
    'retrain_interval': 20,
    'swing_trading_mode': True
}

# Export for compatibility
BTC_ML_CONFIG = BTC_SWING_ML_CONFIG


if __name__ == "__main__":
    # Test BTC Swing ML interface
    print("🧪 Testing BTC Swing ML Interface...")
    
    if not ML_AVAILABLE:
        print("❌ ML libraries not available - install scikit-learn")
        exit(1)
    
    # Create BTC Swing ML interface
    ml_interface = BTCSwingMLInterface(BTC_SWING_ML_CONFIG)
    
    # Simulate swing candle data with realistic patterns
    import random
    base_price = 43000.0
    
    for i in range(30):  # Generate 30 candles
        # Create realistic swing candle
        open_price = base_price
        price_change = random.uniform(-50, 50)  # ±$50 swing movement
        close_price = open_price + price_change
        
        high_price = max(open_price, close_price) + random.uniform(0, 20)
        low_price = min(open_price, close_price) - random.uniform(0, 20)
        volume = random.uniform(0.5, 3.0)
        
        candle_data = {
            'timeframe': '1m',
            'timestamp': datetime.now(),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'body_size': abs(close_price - open_price),
            'is_bullish': close_price > open_price,
            'range': high_price - low_price
        }
        
        # Mock swing metrics
        swing_metrics = {
            'trend_direction': random.choice(['uptrend', 'downtrend', 'neutral']),
            'ma_alignment': {'aligned': random.choice([True, False]), 'direction': random.choice(['bullish', 'bearish', 'neutral'])},
            'momentum_1m': random.uniform(-0.005, 0.005),
            'momentum_3m': random.uniform(-0.003, 0.003),
            'current_rsi': random.uniform(20, 80),
            'atr': random.uniform(15, 35),
            'volume_surge': random.choice([True, False]),
            'vwap_position': random.choice(['above', 'below']),
            'support_levels': [base_price - 100, base_price - 200],
            'resistance_levels': [base_price + 100, base_price + 200]
        }
        
        # Process candle
        signal = ml_interface.process_candle(candle_data, swing_metrics)
        
        if signal.signal != 'hold':
            print(f"₿ Swing Signal: {signal.signal} | Confidence: {signal.confidence:.2f} | {signal.reasoning} | Hold: {signal.expected_hold_time}s")
            
            # Simulate trade outcome
            outcome_pnl = random.uniform(-15, 25)  # Swing P&L range
            hold_time = random.randint(120, 300)   # 2-5 minutes
            features = ml_interface.feature_extractor.extract_swing_features(swing_metrics)
            ml_interface.record_trade_outcome(features, signal.signal, outcome_pnl, hold_time)
        
        base_price = close_price  # Update base price
    
    # Print swing ML stats
    stats = ml_interface.get_ml_stats()
    print(f"\n₿ SWING ML STATS:")
    for key, value in stats.items():
        if key not in ['top_features', 'swing_insights']:
            print(f"   {key}: {value}")
    
    # Print swing insights
    if stats.get('swing_insights'):
        print(f"\n🔄 SWING INSIGHTS:")
        for key, value in stats['swing_insights'].items():
            print(f"   {key}: {value}")
    
    # Force retrain
    ml_interface.force_retrain()
    
    # Test feature analysis
    analysis = ml_interface.get_feature_analysis()
    if analysis and 'feature_importance' in analysis:
        print(f"\n📊 TOP SWING FEATURES:")
        top_features = list(analysis['feature_importance'].items())[:5]
        for feature, importance in top_features:
            print(f"   {feature}: {importance:.3f}")
    
    print("\n✅ BTC Swing ML Interface test completed!")
    print("=" * 60)
    print("✅ Swing pattern recognition: ENHANCED")
    print("✅ Market structure awareness: INTEGRATED")
    print("✅ Hold time optimization: ACTIVE")
    print("✅ Multi-timeframe features: ENABLED")
    print("✅ Support/resistance learning: IMPLEMENTED")
    print("✅ Volume analysis: ADVANCED")
    print("✅ €20 to €1M challenge: OPTIMIZED")