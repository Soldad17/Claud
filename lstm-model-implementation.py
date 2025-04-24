# backend/models/lstm_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os
import logging

logger = logging.getLogger(__name__)

class LSTMModel:
    def __init__(self, config=None):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.config = config or {
            'window_size': 60,
            'lstm_units': 50,
            'dropout_rate': 0.2,
            'dense_units': 25,
            'learning_rate': 0.001,
            'model_path': 'models/lstm_model',
            'batch_size': 32,
            'epochs': 50
        }
        
    def preprocess_data(self, data):
        """Preprocess data for LSTM model"""
        # Ensure data is a DataFrame with 'close' column
        if isinstance(data, list):
            data = pd.DataFrame(data, columns=['close'])
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=['close'])
            
        # Scale the data
        scaled_data = self.scaler.fit_transform(data[['close']])
        
        # Create sequences
        x_data, y_data = [], []
        for i in range(self.config['window_size'], len(scaled_data)):
            x_data.append(scaled_data[i - self.config['window_size']:i, 0])
            y_data.append(scaled_data[i, 0])
            
        return np.array(x_data), np.array(y_data)
    
    def build_model(self):
        """Build LSTM model architecture"""
        model = Sequential()
        model.add(LSTM(units=self.config['lstm_units'], 
                       return_sequences=True, 
                       input_shape=(self.config['window_size'], 1)))
        model.add(Dropout(self.config['dropout_rate']))
        
        model.add(LSTM(units=self.config['lstm_units'], 
                       return_sequences=False))
        model.add(Dropout(self.config['dropout_rate']))
        
        model.add(Dense(units=self.config['dense_units']))
        model.add(Dense(units=1))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        self.model = model
        return model
    
    def train(self, data, validation_split=0.2):
        """Train the LSTM model"""
        # Ensure model is built
        if self.model is None:
            self.build_model()
            
        # Preprocess data
        x_train, y_train = self.preprocess_data(data)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config['model_path']), exist_ok=True)
        
        # Set up callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=self.config['model_path'],
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            x_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Model training completed. Final loss: {history.history['loss'][-1]}")
        return history
    
    def predict(self, data):
        """Generate predictions for given data"""
        if self.model is None:
            logger.error("Model not trained or loaded")
            raise ValueError("Model needs to be trained or loaded before prediction")
            
        # Prepare data for prediction
        if len(data) < self.config['window_size']:
            logger.error(f"Data length {len(data)} is less than window size {self.config['window_size']}")
            raise ValueError(f"Need at least {self.config['window_size']} data points for prediction")
            
        # Scale the data
        scaled_data = self.scaler.transform(np.array(data).reshape(-1, 1))
        
        # Create the data structure for prediction
        x_test = []
        for i in range(self.config['window_size'], len(scaled_data)):
            x_test.append(scaled_data[i - self.config['window_size']:i, 0])
            
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        # Predict
        predictions = self.model.predict(x_test)
        
        # Inverse transform to get actual values
        predictions = self.scaler.inverse_transform(predictions)
        
        # Return predictions with their corresponding timestamps
        return predictions.flatten()
    
    def get_trading_signals(self, closes, threshold=0.01):
        """Generate buy/sell signals based on predictions"""
        # Get predictions
        predictions = self.predict(closes[-self.config['window_size']-10:])
        
        # Calculate percentage change
        last_close = closes[-1]
        predicted_close = predictions[-1]
        percent_change = (predicted_close - last_close) / last_close
        
        # Generate signals based on threshold
        if percent_change > threshold:
            return 'comprar', predicted_close, percent_change
        elif percent_change < -threshold:
            return 'vender', predicted_close, percent_change
        else:
            return 'manter', predicted_close, percent_change
    
    def save(self, path=None):
        """Save model to disk"""
        if self.model is None:
            logger.error("No model to save")
            raise ValueError("Model needs to be trained before saving")
            
        path = path or self.config['model_path']
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_model(self.model, path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path=None):
        """Load model from disk"""
        path = path or self.config['model_path']
        if not os.path.exists(path):
            logger.error(f"Model path {path} does not exist")
            raise FileNotFoundError(f"No model found at {path}")
            
        self.model = load_model(path)
        logger.info(f"Model loaded from {path}")
        return self.model
