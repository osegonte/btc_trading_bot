#!/usr/bin/env python3
"""
Enhanced ML Interface for BTC/USD Trading Bot
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


class BTCMLSignal:
    """Enhanced ML signal structure for BTC"""
    def __init__(self, signal: str, confidence: float, reasoning: str, indicators: Dict = None):
        self.signal = signal        # 'buy', 'sell', 'hold'
        self.confidence = confidence # 0.0 to 1.0
        self.reasoning = reasoning
        self.indicators = indicators or {}


class BTCFeatureExtractor:
    """Extract features from BTC tick data for ML"""
    
    def __init__(self, lookback_ticks: int = 30):
        self.lookback_ticks = lookback_ticks
        self.price_history = deque(maxlen=lookback_ticks)
        self.volume_history = deque(maxlen=lookback_ticks)
        self.spread_history = deque(maxlen=lookback_ticks)
        self.timestamp_history = deque(maxlen=lookback_ticks)
    
    def add_tick(self, tick_data: Dict):
        """Add new BTC tick data"""
        self.price_history.append(tick_data.get('price', 0))
        self.volume_history.append(tick_data.get('size', 1))
        self.spread_history.append(tick_data.get('spread', 1))
        self.timestamp_history.append(tick_data.get('timestamp', datetime.now()))
    
    def extract_features(self) -> Dict:
        """Extract comprehensive features for BTC ML"""
        
        if len(self.price_history) < 15:
            return {}
        
        prices = np.array(list(self.price_history))
        volumes = np.array(list(self.volume_history))
        spreads = np.array(list(self.spread_history))
        
        features = {}
        
        # Price features
        features['current_price'] = prices[-1]
        features['price_change_3'] = (prices[-1] - prices[-3]) / prices[-3] if len(prices) >= 3 and prices[-3] > 0 else 0
        features['price_change_5'] = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 and prices[-5] > 0 else 0
        features['price_change_10'] = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 and prices[-10] > 0 else 0
        features['price_change_15'] = (prices[-1] - prices[-15]) / prices[-15] if len(prices) >= 15 and prices[-15] > 0 else 0
        
        # Volatility features
        features['price_volatility_5'] = np.std(prices[-5:]) / np.mean(prices[-5:]) if len(prices) >= 5 and np.mean(prices[-5:]) > 0 else 0
        features['price_volatility_10'] = np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 and np.mean(prices[-10:]) > 0 else 0
        features['price_volatility_15'] = np.std(prices[-15:]) / np.mean(prices[-15:]) if len(prices) >= 15 and np.mean(prices[-15:]) > 0 else 0
        
        # Moving averages
        features['sma_3'] = np.mean(prices[-3:])
        features['sma_5'] = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
        features['sma_10'] = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
        features['sma_15'] = np.mean(prices[-15:]) if len(prices) >= 15 else prices[-1]
        
        # Price position relative to moving averages
        features['price_above_sma3'] = 1 if prices[-1] > features['sma_3'] else 0
        features['price_above_sma5'] = 1 if prices[-1] > features['sma_5'] else 0
        features['price_above_sma10'] = 1 if prices[-1] > features['sma_10'] else 0
        features['price_above_sma15'] = 1 if prices[-1] > features['sma_15'] else 0
        
        # Moving average crossovers
        features['sma3_above_sma5'] = 1 if features['sma_3'] > features['sma_5'] else 0
        features['sma5_above_sma10'] = 1 if features['sma_5'] > features['sma_10'] else 0
        features['sma10_above_sma15'] = 1 if features['sma_10'] > features['sma_15'] else 0
        
        # Momentum features
        features['momentum_3'] = (prices[-1] - prices[-3]) / prices[-3] if len(prices) >= 3 and prices[-3] > 0 else 0
        features['momentum_5'] = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 and prices[-5] > 0 else 0
        features['momentum_10'] = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 and prices[-10] > 0 else 0
        
        # Volume features
        features['current_volume'] = volumes[-1]
        features['avg_volume_5'] = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
        features['avg_volume_10'] = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
        features['volume_ratio_5'] = volumes[-1] / features['avg_volume_5'] if features['avg_volume_5'] > 0 else 1
        features['volume_ratio_10'] = volumes[-1] / features['avg_volume_10'] if features['avg_volume_10'] > 0 else 1
        
        # Volume trend
        features['volume_trend_5'] = (np.mean(volumes[-3:]) - np.mean(volumes[-8:-3])) if len(volumes) >= 8 else 0
        features['volume_trend_10'] = (np.mean(volumes[-5:]) - np.mean(volumes[-15:-5])) if len(volumes) >= 15 else 0
        
        # Spread features
        features['current_spread'] = spreads[-1]
        features['avg_spread_5'] = np.mean(spreads[-5:]) if len(spreads) >= 5 else spreads[-1]
        features['spread_ratio'] = spreads[-1] / features['avg_spread_5'] if features['avg_spread_5'] > 0 else 1
        
        # Technical indicators
        features['rsi_10'] = self._calculate_rsi(prices, 10)
        features['rsi_14'] = self._calculate_rsi(prices, 14)
        
        # Bollinger band features
        bb_features = self._calculate_bollinger_features(prices, 20)
        features.update(bb_features)
        
        # Price trend features
        features['uptrend_3'] = 1 if self._is_uptrend(prices[-3:]) else 0
        features['uptrend_5'] = 1 if self._is_uptrend(prices[-5:]) else 0
        features['downtrend_3'] = 1 if self._is_downtrend(prices[-3:]) else 0
        features['downtrend_5'] = 1 if self._is_downtrend(prices[-5:]) else 0
        
        # Support/resistance levels
        support_resistance = self._calculate_support_resistance(prices)
        features.update(support_resistance)
        
        return features
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0  # Neutral
        
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
    
    def _calculate_bollinger_features(self, prices: np.ndarray, period: int = 20) -> Dict:
        """Calculate Bollinger Bands features"""
        if len(prices) < period:
            return {
                'bb_upper': prices[-1] * 1.02,
                'bb_middle': prices[-1],
                'bb_lower': prices[-1] * 0.98,
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
    
    def _is_uptrend(self, prices: np.ndarray) -> bool:
        """Check if prices show uptrend"""
        if len(prices) < 2:
            return False
        return prices[-1] > prices[0] and np.mean(np.diff(prices)) > 0
    
    def _is_downtrend(self, prices: np.ndarray) -> bool:
        """Check if prices show downtrend"""
        if len(prices) < 2:
            return False
        return prices[-1] < prices[0] and np.mean(np.diff(prices)) < 0
    
    def _calculate_support_resistance(self, prices: np.ndarray) -> Dict:
        """Calculate basic support/resistance levels"""
        if len(prices) < 10:
            return {
                'near_support': 0,
                'near_resistance': 0,
                'support_strength': 0,
                'resistance_strength': 0
            }
        
        recent_high = np.max(prices[-10:])
        recent_low = np.min(prices[-10:])
        current_price = prices[-1]
        
        # Distance to support/resistance as percentage
        support_distance = (current_price - recent_low) / recent_low if recent_low > 0 else 1
        resistance_distance = (recent_high - current_price) / recent_high if recent_high > 0 else 1
        
        return {
            'near_support': 1 if support_distance < 0.005 else 0,  # Within 0.5%
            'near_resistance': 1 if resistance_distance < 0.005 else 0,
            'support_strength': 1 / (support_distance + 0.001),  # Higher when closer
            'resistance_strength': 1 / (resistance_distance + 0.001)
        }


class BTCMLModel:
    """Enhanced ML model for BTC trading predictions"""
    
    def __init__(self, model_file: str = "btc_trading_model.pkl"):
        self.model_file = model_file
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        
        # Training data storage
        self.training_features = []
        self.training_labels = []
        self.training_outcomes = []  # Store actual P&L for analysis
        
        # Performance tracking
        self.model_version = 1
        self.last_retrain = None
        
        # Load existing model if available
        self.load_model()
    
    def add_training_data(self, features: Dict, outcome: str, profit_loss: float):
        """Add data for BTC model training"""
        
        if not features:
            return
        
        # Convert outcome to label for classification
        if outcome == 'profitable':
            label = 1  # Profitable trade
        elif outcome == 'unprofitable':
            label = 0  # Unprofitable trade
        else:
            return  # Skip no-trade outcomes
        
        # Store training data
        self.training_features.append(features.copy())
        self.training_labels.append(label)
        self.training_outcomes.append(profit_loss)
        
        # Auto-train every 30 samples for BTC (more frequent than gold)
        if len(self.training_features) >= 30 and len(self.training_features) % 15 == 0:
            self.train_model()
    
    def train_model(self, min_samples: int = 25):
        """Train the BTC ML model"""
        
        if not ML_AVAILABLE:
            logging.warning("ML libraries not available for training")
            return False
        
        if len(self.training_features) < min_samples:
            logging.info(f"Need {min_samples} samples to train BTC model. Have {len(self.training_features)}")
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
            
            # Train enhanced model for BTC volatility
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            self.model_version += 1
            self.last_retrain = datetime.now()
            self.save_model()
            
            logging.info(f"✅ BTC ML model trained! Accuracy: {accuracy:.2f} | Version: {self.model_version} | Samples: {len(X_train)}")
            return True
            
        except Exception as e:
            logging.error(f"BTC ML training error: {e}")
            return False
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training arrays for BTC model"""
        
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
    
    def predict(self, features: Dict) -> BTCMLSignal:
        """Make BTC prediction using trained model"""
        
        if not self.is_trained or not self.model:
            return BTCMLSignal('hold', 0.0, 'BTC model not trained', {})
        
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
            
            # Enhanced signal logic for BTC
            signal_indicators = {
                'prediction': prediction,
                'confidence': confidence,
                'model_version': self.model_version,
                'feature_count': len(self.feature_names)
            }
            
            # Convert prediction to signal with BTC-specific thresholds
            if prediction == 1 and confidence > 0.7:  # Higher threshold for BTC
                signal = 'buy'
                reasoning = f'BTC ML: Profitable trade predicted (conf: {confidence:.2f}, v{self.model_version})'
            elif prediction == 0 and confidence > 0.7:
                signal = 'sell'  # Could also be 'hold' depending on strategy
                reasoning = f'BTC ML: Unprofitable avoided (conf: {confidence:.2f}, v{self.model_version})'
            else:
                signal = 'hold'
                reasoning = f'BTC ML: Low confidence {confidence:.2f} (threshold: 0.7)'
            
            return BTCMLSignal(signal, confidence, reasoning, signal_indicators)
            
        except Exception as e:
            logging.error(f"BTC ML prediction error: {e}")
            return BTCMLSignal('hold', 0.0, f'BTC prediction error: {e}', {})
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from BTC model"""
        
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
            logging.error(f"Error getting BTC feature importance: {e}")
            return {}
    
    def save_model(self):
        """Save BTC model to file"""
        
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
                'training_outcomes': self.training_outcomes[-100:]  # Keep last 100 outcomes
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logging.info(f"✅ BTC ML model v{self.model_version} saved to {self.model_file}")
            
        except Exception as e:
            logging.error(f"Error saving BTC model: {e}")
    
    def load_model(self):
        """Load BTC model from file"""
        
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
            
            last_retrain_str = model_data.get('last_retrain')
            if last_retrain_str:
                self.last_retrain = datetime.fromisoformat(last_retrain_str)
            
            logging.info(f"✅ BTC ML model v{self.model_version} loaded from {self.model_file}")
            logging.info(f"   Training samples: {model_data.get('training_samples', 'unknown')}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading BTC model: {e}")
            return False


class BTCMLInterface:
    """Main ML interface for the BTC trading bot"""
    
    def __init__(self, config: Dict = None):
        if config is None:
            config = {}
        
        self.feature_extractor = BTCFeatureExtractor(
            lookback_ticks=config.get('lookback_ticks', 30)
        )
        self.ml_model = BTCMLModel(
            model_file=config.get('model_file', 'btcusd_ml_model.pkl')
        )
        
        # Performance tracking
        self.predictions_made = 0
        self.correct_predictions = 0
        self.total_ml_pnl = 0.0
        
        # BTC-specific tracking
        self.btc_signals_given = 0
        self.successful_btc_signals = 0
        
        logging.info("✅ BTC ML interface initialized")
    
    def process_tick(self, tick_data: Dict) -> BTCMLSignal:
        """Process BTC tick and return ML signal"""
        
        # Add tick to feature extractor
        self.feature_extractor.add_tick(tick_data)
        
        # Extract features
        features = self.feature_extractor.extract_features()
        
        if not features:
            return BTCMLSignal('hold', 0.0, 'Insufficient BTC data for features', {})
        
        # Get ML prediction
        ml_signal = self.ml_model.predict(features)
        
        if ml_signal.signal != 'hold':
            self.predictions_made += 1
            self.btc_signals_given += 1
        
        return ml_signal
    
    def record_trade_outcome(self, features: Dict, signal: str, profit_loss: float):
        """Record BTC trade outcome for model learning"""
        
        if not features:
            return
        
        # Track ML performance
        self.total_ml_pnl += profit_loss
        
        # Determine outcome
        if signal == 'hold':
            outcome = 'no_trade'
        elif profit_loss > 0:
            outcome = 'profitable'
            if self.predictions_made > 0:
                self.correct_predictions += 1
                self.successful_btc_signals += 1
        else:
            outcome = 'unprofitable'
        
        # Add to training data
        self.ml_model.add_training_data(features, outcome, profit_loss)
        
        logging.debug(f"BTC ML outcome recorded: {outcome} | P&L: ${profit_loss:.2f}")
    
    def get_ml_stats(self) -> Dict:
        """Get BTC ML statistics"""
        
        accuracy = (self.correct_predictions / self.predictions_made * 100) if self.predictions_made > 0 else 0
        btc_success_rate = (self.successful_btc_signals / self.btc_signals_given * 100) if self.btc_signals_given > 0 else 0
        feature_importance = self.ml_model.get_feature_importance()
        
        return {
            'ml_available': ML_AVAILABLE,
            'model_trained': self.ml_model.is_trained,
            'model_version': self.ml_model.model_version,
            'training_samples': len(self.ml_model.training_features),
            'predictions_made': self.predictions_made,
            'correct_predictions': self.correct_predictions,
            'accuracy': accuracy,
            'btc_signals_given': self.btc_signals_given,
            'successful_btc_signals': self.successful_btc_signals,
            'btc_success_rate': btc_success_rate,
            'total_ml_pnl': self.total_ml_pnl,
            'avg_ml_pnl': self.total_ml_pnl / max(1, self.predictions_made),
            'last_retrain': self.ml_model.last_retrain.isoformat() if self.ml_model.last_retrain else None,
            'top_features': dict(list(feature_importance.items())[:5])  # Top 5 features
        }
    
    def force_retrain(self):
        """Force BTC model retraining"""
        
        if ML_AVAILABLE:
            success = self.ml_model.train_model(min_samples=15)  # Lower threshold for BTC
            if success:
                logging.info("✅ BTC ML model retrained")
            else:
                logging.warning("⚠️ BTC ML retraining failed")
        else:
            logging.warning("⚠️ ML libraries not available")
    
    def get_feature_analysis(self) -> Dict:
        """Get detailed feature analysis for BTC"""
        
        if not self.ml_model.is_trained:
            return {}
        
        feature_importance = self.ml_model.get_feature_importance()
        current_features = self.feature_extractor.extract_features()
        
        return {
            'feature_importance': feature_importance,
            'current_features': current_features,
            'total_features': len(self.ml_model.feature_names),
            'model_performance': {
                'accuracy': (self.correct_predictions / max(1, self.predictions_made)) * 100,
                'total_predictions': self.predictions_made,
                'model_version': self.ml_model.model_version
            }
        }


# Configuration for BTC ML Interface
BTC_ML_CONFIG = {
    'lookback_ticks': 30,  # Longer lookback for BTC volatility
    'model_file': 'btcusd_ml_model.pkl',
    'min_confidence': 0.70,  # Higher confidence threshold for BTC
    'retrain_interval': 15   # More frequent retraining for crypto
}

# Export for compatibility
ML_CONFIG = BTC_ML_CONFIG
ML_AVAILABLE = True


if __name__ == "__main__":
    # Test BTC ML interface
    print("🤖 Testing BTC ML Interface...")
    
    if not ML_AVAILABLE:
        print("❌ ML libraries not available - install scikit-learn")
        exit(1)
    
    # Create BTC ML interface
    ml_interface = BTCMLInterface(BTC_ML_CONFIG)
    
    # Simulate BTC tick data with realistic volatility
    import random
    base_price = 43000.0
    
    for i in range(50):  # More samples for testing
        # Generate realistic BTC tick
        price_change = random.uniform(-100, 100)  # ±$100 BTC volatility
        base_price += price_change
        base_price = max(30000, min(60000, base_price))  # Keep realistic range
        
        tick_data = {
            'price': base_price,
            'size': random.uniform(0.001, 1.0),  # BTC volume
            'spread': random.uniform(0.5, 3.0),  # BTC spread
            'timestamp': datetime.now()
        }
        
        # Process tick
        signal = ml_interface.process_tick(tick_data)
        
        if signal.signal != 'hold':
            print(f"₿ Signal: {signal.signal} | Confidence: {signal.confidence:.2f} | {signal.reasoning}")
            
            # Simulate trade outcome with BTC-like results
            outcome_pnl = random.uniform(-50, 100)  # BTC P&L range
            features = ml_interface.feature_extractor.extract_features()
            ml_interface.record_trade_outcome(features, signal.signal, outcome_pnl)
    
    # Print BTC ML stats
    stats = ml_interface.get_ml_stats()
    print(f"\n₿ BTC ML Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Force retrain
    ml_interface.force_retrain()
    
    # Test feature analysis
    analysis = ml_interface.get_feature_analysis()
    if analysis:
        print(f"\n📊 Feature Analysis:")
        print(f"   Total features: {analysis['total_features']}")
        print(f"   Top features: {list(analysis['feature_importance'].keys())[:3]}")
    
    print("✅ BTC ML Interface test completed")