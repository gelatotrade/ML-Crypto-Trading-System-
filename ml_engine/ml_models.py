"""
ML Models - LSTM + Ensemble Models for Price Prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for model prediction results"""
    symbol: str
    predicted_return: float
    confidence: float
    direction: str  # 'long', 'short', 'neutral'
    model_predictions: Dict[str, float]


class MLModels:
    """
    Machine Learning models for crypto price prediction.

    Implements:
    - LSTM neural network
    - Random Forest
    - Gradient Boosting
    - Ensemble combination
    """

    def __init__(self, config=None):
        """Initialize ML models"""
        self.config = config
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.is_trained = False

        # Model parameters
        self.sequence_length = 24 if config is None else config.ml.sequence_length
        self.prediction_horizon = 1 if config is None else config.ml.prediction_horizon
        self.min_confidence = 0.6 if config is None else config.ml.min_confidence

    def train(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train all models on the provided data.

        Args:
            features_df: Feature DataFrame
            target: Target returns series
            validation_split: Fraction for validation

        Returns:
            Dictionary of model scores
        """
        # Prepare data
        X, y = self._prepare_data(features_df, target)

        if len(X) < 100:
            logger.warning("Insufficient data for training")
            return {}

        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        scores = {}

        # Train Random Forest
        try:
            rf_score = self._train_random_forest(X_train, y_train, X_val, y_val)
            scores['random_forest'] = rf_score
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")

        # Train Gradient Boosting
        try:
            gb_score = self._train_gradient_boosting(X_train, y_train, X_val, y_val)
            scores['gradient_boosting'] = gb_score
        except Exception as e:
            logger.error(f"Gradient Boosting training failed: {e}")

        # Train LSTM
        try:
            lstm_score = self._train_lstm(X_train, y_train, X_val, y_val)
            scores['lstm'] = lstm_score
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")

        self.is_trained = len(scores) > 0
        logger.info(f"Training complete. Scores: {scores}")

        return scores

    def _prepare_data(
        self,
        features_df: pd.DataFrame,
        target: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Align features and target
        common_idx = features_df.index.intersection(target.index)
        X = features_df.loc[common_idx].values
        y = target.loc[common_idx].values

        # Handle NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]

        # Scale features
        from sklearn.preprocessing import StandardScaler
        self.scalers['features'] = StandardScaler()
        X = self.scalers['features'].fit_transform(X)

        return X, y

    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """Train Random Forest model"""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        self.models['random_forest'] = model

        # Calculate R² score
        score = model.score(X_val, y_val)
        return score

    def _train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """Train Gradient Boosting model"""
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model

        score = model.score(X_val, y_val)
        return score

    def _train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """Train LSTM model"""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            # Reshape for LSTM [samples, timesteps, features]
            X_train_seq = self._create_sequences(X_train)
            y_train_seq = y_train[self.sequence_length-1:]
            X_val_seq = self._create_sequences(X_val)
            y_val_seq = y_val[self.sequence_length-1:]

            if len(X_train_seq) < 10:
                logger.warning("Insufficient sequences for LSTM")
                return 0.0

            # Create model
            input_size = X_train.shape[1]
            hidden_size = 128 if self.config is None else self.config.ml.lstm_hidden_size
            num_layers = 2 if self.config is None else self.config.ml.lstm_num_layers

            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size, hidden_size, num_layers,
                        batch_first=True, dropout=dropout
                    )
                    self.fc = nn.Linear(hidden_size, 1)

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :])

            model = LSTMModel(input_size, hidden_size, num_layers)

            # Training
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            X_tensor = torch.FloatTensor(X_train_seq)
            y_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1)

            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            model.train()
            for epoch in range(50):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            self.models['lstm'] = model

            # Validation
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val_seq)
                y_val_tensor = torch.FloatTensor(y_val_seq)
                predictions = model(X_val_tensor).squeeze()
                mse = ((predictions - y_val_tensor) ** 2).mean().item()
                var = y_val_tensor.var().item()
                score = 1 - mse / (var + 1e-10)

            return score

        except ImportError:
            logger.warning("PyTorch not available, skipping LSTM")
            return 0.0

    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM"""
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
        return np.array(sequences)

    def predict(
        self,
        features_df: pd.DataFrame,
        symbol: str = ""
    ) -> PredictionResult:
        """
        Generate predictions using ensemble of models.

        Args:
            features_df: Feature DataFrame for prediction
            symbol: Symbol being predicted

        Returns:
            PredictionResult with ensemble prediction
        """
        if not self.is_trained:
            return PredictionResult(
                symbol=symbol,
                predicted_return=0.0,
                confidence=0.0,
                direction='neutral',
                model_predictions={}
            )

        # Prepare features
        X = features_df.values[-1:] if len(features_df) > 0 else np.array([[]])

        if 'features' in self.scalers and X.size > 0:
            try:
                X = self.scalers['features'].transform(X)
            except Exception:
                pass

        predictions = {}

        # Random Forest prediction
        if 'random_forest' in self.models:
            try:
                predictions['random_forest'] = self.models['random_forest'].predict(X)[0]
            except Exception as e:
                logger.debug(f"RF prediction failed: {e}")

        # Gradient Boosting prediction
        if 'gradient_boosting' in self.models:
            try:
                predictions['gradient_boosting'] = self.models['gradient_boosting'].predict(X)[0]
            except Exception as e:
                logger.debug(f"GB prediction failed: {e}")

        # LSTM prediction
        if 'lstm' in self.models:
            try:
                import torch
                X_seq = self._create_sequences(
                    self.scalers['features'].transform(features_df.values)
                )
                if len(X_seq) > 0:
                    self.models['lstm'].eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_seq[-1:])
                        predictions['lstm'] = self.models['lstm'](X_tensor).item()
            except Exception as e:
                logger.debug(f"LSTM prediction failed: {e}")

        if not predictions:
            return PredictionResult(
                symbol=symbol,
                predicted_return=0.0,
                confidence=0.0,
                direction='neutral',
                model_predictions={}
            )

        # Ensemble: weighted average
        weights = {'random_forest': 0.3, 'gradient_boosting': 0.3, 'lstm': 0.4}
        total_weight = sum(weights.get(k, 0) for k in predictions)

        if total_weight > 0:
            ensemble_pred = sum(
                predictions[k] * weights.get(k, 0) for k in predictions
            ) / total_weight
        else:
            ensemble_pred = np.mean(list(predictions.values()))

        # Calculate confidence based on model agreement
        if len(predictions) > 1:
            pred_std = np.std(list(predictions.values()))
            pred_mean = abs(np.mean(list(predictions.values())))
            confidence = max(0, 1 - pred_std / (pred_mean + 1e-10))
        else:
            confidence = 0.5

        # Determine direction
        if ensemble_pred > 0.001 and confidence >= self.min_confidence:
            direction = 'long'
        elif ensemble_pred < -0.001 and confidence >= self.min_confidence:
            direction = 'short'
        else:
            direction = 'neutral'

        return PredictionResult(
            symbol=symbol,
            predicted_return=ensemble_pred,
            confidence=confidence,
            direction=direction,
            model_predictions=predictions
        )

    def save_models(self, path: str = "models/"):
        """Save trained models to disk"""
        os.makedirs(path, exist_ok=True)

        for name, model in self.models.items():
            if name == 'lstm':
                try:
                    import torch
                    torch.save(model.state_dict(), f"{path}/{name}_model.pt")
                except Exception:
                    pass
            else:
                with open(f"{path}/{name}_model.pkl", 'wb') as f:
                    pickle.dump(model, f)

        # Save scalers
        with open(f"{path}/scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)

        logger.info(f"Models saved to {path}")

    def load_models(self, path: str = "models/"):
        """Load trained models from disk"""
        if not os.path.exists(path):
            logger.warning(f"Model path {path} does not exist")
            return

        # Load scalers
        scaler_path = f"{path}/scalers.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scalers = pickle.load(f)

        # Load sklearn models
        for name in ['random_forest', 'gradient_boosting']:
            model_path = f"{path}/{name}_model.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[name] = pickle.load(f)

        self.is_trained = len(self.models) > 0
        logger.info(f"Loaded {len(self.models)} models from {path}")
