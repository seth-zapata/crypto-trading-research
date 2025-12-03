# Cryptocurrency Trading System - Implementation Specification (Part 2)
## Model Development, RL Agents, Risk Management & Deployment

**Continuation from Part 1**

This document covers Phases 3-6: the intelligence layer (ML models), decision layer (RL agents), risk/execution systems, and production deployment.

---

## Phase 3: Model Development (Week 3-4)

### 3.1 LightGBM Baseline (Start Here)

```python
# models/predictors/lightgbm_baseline.py

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple

class LightGBMPredictor:
    """
    Gradient boosting baseline - fast, interpretable, strong performance
    
    Research shows: Properly tuned GBMs often outperform deep learning
    for tabular financial data with lower computational cost
    
    Target: Binary classification (price up/down in next period)
    Alternative: Regression (predict next period return)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.feature_importance = None
        
    def prepare_data(self, 
                    features: pd.DataFrame,
                    target_col: str = 'return_1',
                    horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for supervised learning
        
        Args:
            features: DataFrame with engineered features
            target_col: Column to predict
            horizon: Periods ahead to predict (1 = next period)
        """
        # Create target (future returns)
        y = features[target_col].shift(-horizon)
        
        # Convert to binary classification (up/down)
        y_binary = (y > 0).astype(int)
        
        # Remove NaN rows
        valid_idx = ~(y_binary.isna() | features.isna().any(axis=1))
        
        X = features[valid_idx]
        y = y_binary[valid_idx]
        
        return X, y
    
    def train_with_cv(self,
                     X: pd.DataFrame,
                     y: pd.Series,
                     n_splits: int = 5) -> Dict:
        """
        Train with time-series cross-validation
        
        Critical: Use TimeSeriesSplit to avoid look-ahead bias
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        cv_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"Training fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                self.config,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            # Predict on validation
            val_preds = model.predict(X_val)
            val_preds_binary = (val_preds > 0.5).astype(int)
            
            # Calculate accuracy
            accuracy = (val_preds_binary == y_val).mean()
            cv_scores.append(accuracy)
            
            print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
        
        # Train final model on all data
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            self.config,
            train_data,
            num_boost_round=1000
        )
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print(f"\nCross-validation Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return {
            'cv_scores': cv_scores,
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'feature_importance': self.feature_importance
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_with_cv() first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores
        
        Confidence = absolute distance from 0.5 (decision boundary)
        """
        probs = self.predict(X)
        predictions = (probs > 0.5).astype(int)
        confidence = np.abs(probs - 0.5) * 2  # Scale to [0, 1]
        
        return predictions, confidence
    
    def save_model(self, path: str):
        """Save trained model"""
        self.model.save_model(path)
    
    def load_model(self, path: str):
        """Load trained model"""
        self.model = lgb.Booster(model_file=path)

# Example usage:
config = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Load features
features = load_features_from_db('BTC/USDT', start='2023-01-01', end='2024-12-01')

# Initialize predictor
predictor = LightGBMPredictor(config)

# Prepare data
X, y = predictor.prepare_data(features, target_col='return_1', horizon=1)

# Train with cross-validation
results = predictor.train_with_cv(X, y, n_splits=5)

# Show top features
print("\nTop 20 Important Features:")
print(results['feature_importance'].head(20))

# Save model
predictor.save_model('models/saved/lightgbm_btc_1h.txt')

# Predict on new data
new_features = load_latest_features()
predictions, confidence = predictor.predict_with_confidence(new_features)
```

### 3.2 LSTM Sequential Model

```python
# models/predictors/lstm_sequential.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series with sliding windows
    """
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int = 60):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        X = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        
        return torch.FloatTensor(X), torch.FloatTensor([y])

class LSTMPredictor(nn.Module):
    """
    LSTM model for time series prediction
    
    Architecture:
    - Input: (batch, sequence_length, num_features)
    - LSTM layers with dropout
    - Fully connected output layer
    
    Research: LSTM excels at capturing temporal dependencies in financial data
    R² of ~11.5% on order book prediction (OpenReview 2024)
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 1):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take output from last time step
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        dropped = self.dropout(last_output)
        
        # Fully connected
        output = self.fc(dropped)
        
        # Sigmoid activation for binary classification
        output = self.sigmoid(output)
        
        return output

class LSTMTrainer:
    """
    Training pipeline for LSTM models
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, 
                   train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, 
                val_loader: DataLoader,
                criterion: nn.Module) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = (outputs > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def fit(self,
           train_loader: DataLoader,
           val_loader: DataLoader,
           num_epochs: int = 100,
           learning_rate: float = 0.001,
           patience: int = 10) -> Dict:
        """
        Full training loop with early stopping
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_accuracy = self.validate(val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/saved/lstm_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('models/saved/lstm_best.pth'))
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }

# Example usage:
# Load and prepare data
features = load_features_from_db('BTC/USDT')
X, y = prepare_lstm_data(features)

# Split into train/val
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Create datasets and loaders
train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length=60)
val_dataset = TimeSeriesDataset(X_val, y_val, sequence_length=60)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
input_size = X_train.shape[1]  # Number of features
model = LSTMPredictor(input_size=input_size, hidden_size=128, num_layers=2)

# Train
trainer = LSTMTrainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')
results = trainer.fit(train_loader, val_loader, num_epochs=100, learning_rate=0.001)
```

### 3.3 Graph Neural Network (GNN)

```python
# models/predictors/gnn_crossasset.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import List

class GNNCrossAssetPredictor(nn.Module):
    """
    Graph Neural Network for multi-asset prediction
    
    Architecture:
    - Graph Attention Networks (GAT) for learning edge importance
    - Multiple GNN layers for information propagation
    - Node-level predictions for each asset
    
    Research validation:
    - THGNN: 2.2x Sharpe improvement
    - MGAR: 164-236% returns with multi-view graphs
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        super(GNNCrossAssetPredictor, self).__init__()
        
        self.num_layers = num_layers
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, 
                       heads=num_heads, dropout=dropout)
            )
        
        # Output layer (single head)
        self.gat_layers.append(
            GATConv(hidden_dim * num_heads, hidden_dim, 
                   heads=1, dropout=dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Final prediction layer
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features (optional)
            batch: Batch assignment for multiple graphs
        """
        # Apply GAT layers
        for i, gat in enumerate(self.gat_layers[:-1]):
            x = gat(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
        
        # Final GAT layer
        x = self.gat_layers[-1](x, edge_index)
        
        # Node-level predictions
        x = self.fc_out(x)
        x = self.sigmoid(x)
        
        return x

class GNNTrainer:
    """
    Training pipeline for GNN models
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        
    def train_epoch(self,
                   graphs: List[Data],
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module) -> float:
        """Train on batch of graphs"""
        self.model.train()
        total_loss = 0
        
        for graph in graphs:
            graph = graph.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            out = self.model(graph.x, graph.edge_index, graph.edge_attr)
            
            # Loss (compare predictions to targets)
            loss = criterion(out, graph.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(graphs)
    
    def validate(self, 
                graphs: List[Data],
                criterion: nn.Module) -> Tuple[float, float]:
        """Validate on graphs"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for graph in graphs:
                graph = graph.to(self.device)
                
                out = self.model(graph.x, graph.edge_index, graph.edge_attr)
                loss = criterion(out, graph.y)
                
                total_loss += loss.item()
                
                # Accuracy
                predictions = (out > 0.5).float()
                correct += (predictions == graph.y).sum().item()
                total += graph.y.size(0)
        
        avg_loss = total_loss / len(graphs)
        accuracy = correct / total
        
        return avg_loss, accuracy

# Example usage:
# Build graphs
prices = load_prices(['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
builder = CryptoGraphBuilder(threshold=0.3)
graphs = builder.build_temporal_correlation_graph(prices, window=20, num_snapshots=100)

# Add targets to graphs
for i, graph in enumerate(graphs):
    # Target: next period returns for each asset
    future_returns = calculate_future_returns(prices, i)
    graph.y = torch.tensor((future_returns > 0).astype(float).values, dtype=torch.float).view(-1, 1)

# Split train/val
split_idx = int(len(graphs) * 0.8)
train_graphs = graphs[:split_idx]
val_graphs = graphs[split_idx:]

# Initialize model
input_dim = graphs[0].x.shape[1]
model = GNNCrossAssetPredictor(input_dim=input_dim, hidden_dim=64, num_layers=3)

# Train
trainer = GNNTrainer(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(100):
    train_loss = trainer.train_epoch(train_graphs, optimizer, criterion)
    val_loss, val_acc = trainer.validate(val_graphs, criterion)
    
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
```

### 3.4 Ensemble Predictor

```python
# models/predictors/ensemble.py

import numpy as np
import pandas as pd
from typing import Dict, List

class EnsemblePredictor:
    """
    Combine predictions from multiple models
    
    Weighting strategies:
    - Fixed: Predetermined weights
    - Performance-based: Weight by historical accuracy
    - Confidence-weighted: Weight by prediction confidence
    
    Research: Ensemble typically improves by 2-5% over best individual model
    """
    
    def __init__(self, 
                 models: Dict[str, object],
                 weights: Dict[str, float],
                 min_confidence: float = 0.6):
        self.models = models
        self.weights = weights
        self.min_confidence = min_confidence
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble predictions
        
        Returns:
            predictions: Binary predictions (0/1)
            confidence: Ensemble confidence scores
        """
        all_predictions = {}
        all_confidences = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if name == 'lightgbm':
                probs = model.predict(X)
                preds = (probs > 0.5).astype(int)
                conf = np.abs(probs - 0.5) * 2
                
            elif name == 'lstm':
                # LSTM expects 3D input [batch, sequence, features]
                # Need to reshape X appropriately
                X_lstm = prepare_lstm_sequences(X)
                with torch.no_grad():
                    probs = model(torch.FloatTensor(X_lstm)).numpy()
                preds = (probs > 0.5).astype(int)
                conf = np.abs(probs - 0.5) * 2
                
            elif name == 'gnn':
                # GNN expects graph structure
                graph = convert_to_graph(X)
                with torch.no_grad():
                    probs = model(graph.x, graph.edge_index).numpy()
                preds = (probs > 0.5).astype(int)
                conf = np.abs(probs - 0.5) * 2
            
            all_predictions[name] = preds
            all_confidences[name] = conf
        
        # Weighted voting
        weighted_sum = np.zeros(len(X))
        for name, preds in all_predictions.items():
            weighted_sum += preds.flatten() * self.weights[name]
        
        ensemble_predictions = (weighted_sum > 0.5).astype(int)
        
        # Weighted confidence
        ensemble_confidence = np.zeros(len(X))
        for name, conf in all_confidences.items():
            ensemble_confidence += conf.flatten() * self.weights[name]
        
        return ensemble_predictions, ensemble_confidence
    
    def predict_with_threshold(self, 
                              X: pd.DataFrame,
                              threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Only make predictions when confidence exceeds threshold
        
        Low confidence predictions → neutral (no action)
        """
        if threshold is None:
            threshold = self.min_confidence
        
        predictions, confidence = self.predict(X)
        
        # Mask low-confidence predictions
        high_confidence_mask = confidence >= threshold
        
        # Set low confidence to neutral (-1)
        filtered_predictions = np.full(len(predictions), -1)
        filtered_predictions[high_confidence_mask] = predictions[high_confidence_mask]
        
        return filtered_predictions, confidence

# Example usage:
# Load trained models
lightgbm_model = load_lightgbm_model('models/saved/lightgbm_btc_1h.txt')
lstm_model = load_lstm_model('models/saved/lstm_best.pth')
gnn_model = load_gnn_model('models/saved/gnn_best.pth')

# Create ensemble
ensemble = EnsemblePredictor(
    models={
        'lightgbm': lightgbm_model,
        'lstm': lstm_model,
        'gnn': gnn_model
    },
    weights={
        'lightgbm': 0.4,
        'lstm': 0.35,
        'gnn': 0.25
    },
    min_confidence=0.6
)

# Generate predictions
features = load_latest_features()
predictions, confidence = ensemble.predict_with_threshold(features, threshold=0.65)

# Interpret:
# -1: No signal (low confidence)
#  0: Bearish (high confidence)
#  1: Bullish (high confidence)
```

---

## Phase 4: Reinforcement Learning Agents (Week 4-5)

### 4.1 Custom Trading Environment

```python
# agents/environments/crypto_env.py

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class CryptoTradingEnv(gym.Env):
    """
    Custom OpenAI Gym environment for crypto trading
    
    Complies with FinRL framework specifications
    
    State space:
    - Market features (prices, indicators, on-chain metrics)
    - Position info (size, P&L, duration)
    - Risk metrics (exposure, drawdown)
    
    Action space:
    - Continuous: [-1, 1] position size
    - Alternative: Discrete {strong_sell, sell, hold, buy, strong_buy}
    
    Reward:
    - Primary: Risk-adjusted returns
    - Penalties: Drawdowns, excessive trading
    - Bonuses: Profit-taking, risk management
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 df: pd.DataFrame,
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.001,
                 reward_scaling: float = 1e4):
        super(CryptoTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        
        # Action space: continuous position sizing
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(1,),
            dtype=np.float32
        )
        
        # State space: market features + position state
        num_features = len(self.df.columns) - 1  # Exclude time column
        position_features = 5  # position, entry_price, unrealized_pnl, steps_held, drawdown
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features + position_features,),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # Current position size (-1 to 1)
        self.entry_price = 0.0
        self.peak_balance = self.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        
        self.portfolio_history = [self.initial_balance]
        self.trade_history = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Construct state vector"""
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape[0])
        
        # Market features
        market_row = self.df.iloc[self.current_step]
        market_features = market_row.drop('time').values
        
        # Position features
        current_price = market_row['close']
        
        if self.position != 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * self.position
        else:
            unrealized_pnl = 0.0
        
        steps_held = self.total_trades if self.position != 0 else 0
        current_drawdown = (self.balance - self.peak_balance) / self.peak_balance
        
        position_features = np.array([
            self.position,
            unrealized_pnl,
            self.balance / self.initial_balance - 1,  # Total return
            steps_held,
            current_drawdown
        ])
        
        # Combine
        obs = np.concatenate([market_features, position_features])
        
        return obs.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state"""
        action = np.clip(action[0], -1, 1)
        
        current_price = self.df.iloc[self.current_step]['close']
        
        # Calculate position change
        position_change = action - self.position
        
        # Execute trade if significant change
        if abs(position_change) > 0.05:  # 5% threshold
            trade_value = abs(position_change) * self.balance
            cost = trade_value * self.transaction_cost
            
            self.balance -= cost
            self.position = action
            
            if self.position != 0:
                self.entry_price = current_price
            
            self.total_trades += 1
            
            self.trade_history.append({
                'step': self.current_step,
                'action': action,
                'price': current_price,
                'cost': cost
            })
        
        # Update P&L
        if self.position != 0:
            price_change = current_price - self.df.iloc[self.current_step - 1]['close']
            pnl = price_change / self.df.iloc[self.current_step - 1]['close'] * self.position * self.balance
            self.balance += pnl
        
        # Update portfolio value
        self.peak_balance = max(self.peak_balance, self.balance)
        self.portfolio_history.append(self.balance)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades
        }
        
        return obs, reward, done, info
    
    def _calculate_reward(self) -> float:
        """
        Multi-component reward function
        
        Components:
        1. Returns (primary)
        2. Sharpe ratio (risk-adjusted)
        3. Drawdown penalty
        4. Trading cost penalty
        5. Consistency bonus
        """
        if len(self.portfolio_history) < 2:
            return 0.0
        
        # Portfolio return
        portfolio_return = (self.portfolio_history[-1] - self.portfolio_history[-2]) / self.portfolio_history[-2]
        
        # Sharpe ratio component (if sufficient history)
        if len(self.portfolio_history) > 30:
            returns = pd.Series(self.portfolio_history).pct_change().dropna()
            sharpe = returns.mean() / (returns.std() + 1e-6) * np.sqrt(252)
            sharpe_reward = sharpe * 0.01
        else:
            sharpe_reward = 0.0
        
        # Drawdown penalty
        current_drawdown = (self.balance - self.peak_balance) / self.peak_balance
        drawdown_penalty = current_drawdown * 0.5 if current_drawdown < -0.05 else 0.0
        
        # Overtrading penalty
        if self.current_step > 0:
            trade_frequency = self.total_trades / self.current_step
            overtrading_penalty = -max(0, trade_frequency - 0.1) * 0.1
        else:
            overtrading_penalty = 0.0
        
        # Total reward
        reward = (
            portfolio_return * 100 +     # Main component (scaled)
            sharpe_reward +               # Risk adjustment
            drawdown_penalty +            # Drawdown penalty
            overtrading_penalty           # Trading frequency
        ) * self.reward_scaling
        
        return reward
    
    def render(self, mode='human'):
        """Render environment state"""
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position: {self.position:.2f}")
        print(f"Total Return: {(self.balance / self.initial_balance - 1) * 100:.2f}%")
        print(f"Total Trades: {self.total_trades}")
        print("---")

# Example usage:
# Load data
df = load_features_from_db('BTC/USDT', start='2023-01-01', end='2024-12-01')

# Create environment
env = CryptoTradingEnv(df, initial_balance=10000, transaction_cost=0.001)

# Test environment
obs = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    
    if done:
        break

print(f"Final Balance: ${env.balance:.2f}")
print(f"Total Return: {(env.balance / env.initial_balance - 1) * 100:.2f}%")
```

### 4.2 Hierarchical RL System

```python
# agents/hierarchical/hierarchical_system.py

import torch
from stable_baselines3 import PPO, DDPG, DQN
from typing import Dict, List
import numpy as np

class HierarchicalTradingSystem:
    """
    Three-level hierarchical RL system
    
    Levels:
    1. Strategic (weekly): Asset selection, regime detection (PPO)
    2. Tactical (daily): Position sizing, entry/exit timing (DDPG)
    3. Execution (minutes): Order placement, slippage minimization (DQN)
    
    Research validation: HRT paper shows Sharpe 2.74 vs 2.27 benchmark
    """
    
    def __init__(self):
        self.strategic_agent = None
        self.tactical_agent = None
        self.execution_agent = None
        
    def train_strategic_agent(self,
                             env: gym.Env,
                             total_timesteps: int = 100000):
        """
        Train high-level agent for asset selection
        
        State: Macro indicators, on-chain regime, cross-asset correlations
        Action: Portfolio allocation weights
        """
        self.strategic_agent = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,  # Long-term focus
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./logs/strategic/"
        )
        
        print("Training strategic agent...")
        self.strategic_agent.learn(total_timesteps=total_timesteps)
        self.strategic_agent.save("models/saved/strategic_agent")
        
    def train_tactical_agent(self,
                            env: gym.Env,
                            total_timesteps: int = 100000):
        """
        Train mid-level agent for position sizing and timing
        
        State: Technical indicators, ML predictions, current position
        Action: Position size adjustment (-1 to 1)
        """
        self.tactical_agent = DDPG(
            'MlpPolicy',
            env,
            learning_rate=1e-3,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=128,
            tau=0.005,
            gamma=0.95,  # Medium-term focus
            verbose=1,
            tensorboard_log="./logs/tactical/"
        )
        
        print("Training tactical agent...")
        self.tactical_agent.learn(total_timesteps=total_timesteps)
        self.tactical_agent.save("models/saved/tactical_agent")
    
    def train_execution_agent(self,
                             env: gym.Env,
                             total_timesteps: int = 50000):
        """
        Train low-level agent for order execution
        
        State: Order book, recent trades, market depth
        Action: Order type (market/limit), timing
        """
        self.execution_agent = DQN(
            'MlpPolicy',
            env,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.9,  # Short-term focus
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            verbose=1,
            tensorboard_log="./logs/execution/"
        )
        
        print("Training execution agent...")
        self.execution_agent.learn(total_timesteps=total_timesteps)
        self.execution_agent.save("models/saved/execution_agent")
    
    def predict_hierarchical(self, 
                            strategic_obs: np.ndarray,
                            tactical_obs: np.ndarray,
                            execution_obs: np.ndarray) -> Dict:
        """
        Generate decisions from all three levels
        
        Flow:
        1. Strategic: Which assets, how much capital per asset
        2. Tactical: When to enter/exit, position size
        3. Execution: How to place orders
        """
        # Strategic decision
        strategic_action, _ = self.strategic_agent.predict(strategic_obs, deterministic=True)
        
        # Tactical decision (conditioned on strategic)
        tactical_action, _ = self.tactical_agent.predict(tactical_obs, deterministic=True)
        
        # Execution decision (conditioned on tactical)
        execution_action, _ = self.execution_agent.predict(execution_obs, deterministic=True)
        
        return {
            'strategic': strategic_action,
            'tactical': tactical_action,
            'execution': execution_action
        }
    
    def load_agents(self):
        """Load pre-trained agents"""
        self.strategic_agent = PPO.load("models/saved/strategic_agent")
        self.tactical_agent = DDPG.load("models/saved/tactical_agent")
        self.execution_agent = DQN.load("models/saved/execution_agent")

# Example usage:
# Create environments for each level
strategic_env = create_strategic_env()  # Weekly rebalancing
tactical_env = create_tactical_env()    # Daily trading
execution_env = create_execution_env()  # Minute-level orders

# Initialize hierarchical system
hs = HierarchicalTradingSystem()

# Train agents sequentially
hs.train_strategic_agent(strategic_env, total_timesteps=100000)
hs.train_tactical_agent(tactical_env, total_timesteps=100000)
hs.train_execution_agent(execution_env, total_timesteps=50000)

# Use in production
hs.load_agents()
decisions = hs.predict_hierarchical(strategic_obs, tactical_obs, execution_obs)
```

---

## Phase 5: Risk Management & Execution (Week 5-6)

### 5.1 Position Sizing & Risk Management

```python
# risk/position_sizer.py

import numpy as np
import pandas as pd
from typing import Tuple

class PositionSizer:
    """
    Calculate optimal position sizes using Kelly Criterion
    
    Methods:
    - Full Kelly: Theoretical maximum
    - Half Kelly: Practical (safer)
    - Fixed fractional: Constant risk per trade
    
    Research: Half-Kelly provides best risk-adjusted returns
    """
    
    def __init__(self, method: str = 'half_kelly'):
        self.method = method
        
    def calculate_kelly(self,
                       win_rate: float,
                       avg_win: float,
                       avg_loss: float) -> float:
        """
        Kelly formula: f = (p * W - (1-p)) / W
        
        Where:
        - p = win probability
        - W = average win / average loss ratio
        - f = fraction of capital to risk
        """
        if avg_loss == 0:
            return 0.0
        
        W = avg_win / avg_loss
        kelly = (win_rate * W - (1 - win_rate)) / W
        
        # Clip to reasonable range
        kelly = np.clip(kelly, 0, 0.25)  # Max 25% of capital
        
        if self.method == 'half_kelly':
            kelly = kelly * 0.5
        elif self.method == 'quarter_kelly':
            kelly = kelly * 0.25
        
        return kelly
    
    def size_position(self,
                     signal_confidence: float,
                     account_balance: float,
                     entry_price: float,
                     stop_loss: float,
                     volatility: float,
                     historical_performance: Dict) -> float:
        """
        Calculate position size incorporating multiple factors
        
        Args:
            signal_confidence: Model confidence (0-1)
            account_balance: Available capital
            entry_price: Intended entry price
            stop_loss: Stop loss price
            volatility: Asset volatility
            historical_performance: Past win rate, avg win/loss
        
        Returns:
            position_size: Quantity to trade
        """
        # Extract historical metrics
        win_rate = historical_performance.get('win_rate', 0.5)
        avg_win = historical_performance.get('avg_win', 0.02)
        avg_loss = historical_performance.get('avg_loss', 0.01)
        
        # Calculate Kelly fraction
        kelly_fraction = self.calculate_kelly(win_rate, avg_win, avg_loss)
        
        # Adjust for confidence
        adjusted_fraction = kelly_fraction * signal_confidence
        
        # Calculate risk amount
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0.0
        
        # Position size based on risk
        max_risk_amount = account_balance * adjusted_fraction
        position_size = max_risk_amount / risk_per_share
        
        # Volatility adjustment (reduce size in high volatility)
        vol_adjustment = min(1.0, 0.20 / volatility)  # Target 20% annual vol
        position_size *= vol_adjustment
        
        # Ensure position value doesn't exceed fraction of capital
        max_position_value = account_balance * 0.25  # Max 25% per position
        max_size_by_value = max_position_value / entry_price
        
        position_size = min(position_size, max_size_by_value)
        
        return position_size

class RiskManager:
    """
    Enforce risk limits across portfolio
    
    Checks:
    - Maximum position size
    - Maximum leverage
    - Maximum drawdown
    - Daily loss limit
    - Correlation limits
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.peak_equity = 0
        self.daily_pnl = []
        
    def check_position_allowed(self,
                              new_position: Dict,
                              current_positions: List[Dict],
                              account_balance: float) -> Tuple[bool, str]:
        """
        Validate new position against risk limits
        
        Returns: (allowed, reason)
        """
        # Position size limit
        max_position_pct = self.config['max_position_size']
        position_value = new_position['size'] * new_position['price']
        position_pct = position_value / account_balance
        
        if position_pct > max_position_pct:
            return False, f"Position size {position_pct:.1%} exceeds limit {max_position_pct:.1%}"
        
        # Leverage limit
        total_exposure = sum(p['size'] * p['price'] for p in current_positions)
        new_exposure = total_exposure + position_value
        leverage = new_exposure / account_balance
        
        if leverage > self.config['max_leverage']:
            return False, f"Leverage {leverage:.2f}x exceeds limit {self.config['max_leverage']}x"
        
        # Drawdown limit
        self.peak_equity = max(self.peak_equity, account_balance)
        drawdown = (account_balance - self.peak_equity) / self.peak_equity
        
        if drawdown < -self.config['max_drawdown']:
            return False, f"Drawdown {drawdown:.1%} exceeds limit"
        
        # Daily loss limit
        if len(self.daily_pnl) > 0:
            daily_loss = sum(self.daily_pnl[-24:]) / account_balance  # Last 24 hours
            if daily_loss < -self.config['max_daily_loss']:
                return False, f"Daily loss {daily_loss:.1%} exceeds limit"
        
        # Correlation limit (avoid over-concentration)
        same_sector_positions = [p for p in current_positions 
                                if p.get('sector') == new_position.get('sector')]
        same_sector_exposure = sum(p['size'] * p['price'] for p in same_sector_positions)
        
        if (same_sector_exposure + position_value) / account_balance > self.config['max_correlated_positions']:
            return False, "Too much exposure to correlated assets"
        
        return True, "OK"

# Example usage:
position_sizer = PositionSizer(method='half_kelly')

# Calculate position size
size = position_sizer.size_position(
    signal_confidence=0.75,
    account_balance=10000,
    entry_price=50000,
    stop_loss=49000,
    volatility=0.60,
    historical_performance={
        'win_rate': 0.55,
        'avg_win': 0.025,
        'avg_loss': 0.015
    }
)

print(f"Position size: {size:.4f} BTC")
print(f"Position value: ${size * 50000:.2f}")
```

---

## Phase 6: Deployment & Monitoring (Week 6)

### 6.1 Main Orchestrator

```python
# orchestrator.py

import asyncio
import logging
from datetime import datetime
from typing import Dict, List

class TradingOrchestrator:
    """
    Main system coordinator
    
    Responsibilities:
    - Fetch data from all sources
    - Generate predictions from ensemble
    - Get RL agent decisions
    - Apply risk checks
    - Execute trades
    - Monitor performance
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        self.data_ingester = ExchangeDataIngester(config['exchanges'])
        self.onchain_provider = OnChainDataProvider(config['onchain_api_key'])
        self.sentiment_analyzer = SentimentAnalyzer(config['sentiment'])
        
        self.feature_engineer = FeatureEngineer()
        self.ensemble = load_ensemble_models()
        self.rl_system = HierarchicalTradingSystem()
        self.rl_system.load_agents()
        
        self.position_sizer = PositionSizer(method='half_kelly')
        self.risk_manager = RiskManager(config['risk'])
        
        self.execution = ExecutionEngine(config['exchanges'])
        self.portfolio = Portfolio(initial_capital=config['initial_capital'])
        
        self.running = False
        
    async def main_loop(self):
        """
        Main trading loop
        
        Frequency: Every 1 hour (adjustable)
        """
        self.running = True
        logging.info("Starting trading orchestrator...")
        
        while self.running:
            try:
                # 1. Fetch latest data
                market_data = await self._fetch_market_data()
                onchain_data = await self._fetch_onchain_data()
                sentiment_data = await self._fetch_sentiment_data()
                
                # 2. Generate features
                features = self.feature_engineer.generate_all_features(market_data)
                
                # 3. Get regime classification from on-chain
                regime = self.onchain_provider.get_regime_classification(onchain_data)
                logging.info(f"Current regime: {regime}")
                
                # 4. Check circuit breaker
                if self._should_halt_trading(market_data, self.portfolio.balance):
                    logging.warning("Circuit breaker triggered - pausing trading")
                    await asyncio.sleep(3600)
                    continue
                
                # 5. Get ML predictions
                ml_predictions, ml_confidence = self.ensemble.predict_with_threshold(features)
                
                # 6. Get RL decisions
                strategic_obs = self._prepare_strategic_obs(onchain_data, regime)
                tactical_obs = self._prepare_tactical_obs(features, ml_predictions)
                execution_obs = self._prepare_execution_obs(market_data)
                
                rl_decisions = self.rl_system.predict_hierarchical(
                    strategic_obs, tactical_obs, execution_obs
                )
                
                # 7. Combine signals
                final_decisions = self._combine_signals(
                    ml_predictions,
                    ml_confidence,
                    rl_decisions,
                    sentiment_data,
                    regime
                )
                
                # 8. Execute decisions
                for decision in final_decisions:
                    await self._execute_decision(decision)
                
                # 9. Update portfolio state
                self.portfolio.update(market_data)
                
                # 10. Log metrics
                self._log_performance()
                
                # Sleep until next iteration
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logging.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _execute_decision(self, decision: Dict):
        """
        Execute a trading decision with risk checks
        """
        # Check risk limits
        allowed, reason = self.risk_manager.check_position_allowed(
            decision,
            self.portfolio.positions,
            self.portfolio.balance
        )
        
        if not allowed:
            logging.warning(f"Trade rejected: {reason}")
            return
        
        # Calculate position size
        size = self.position_sizer.size_position(
            signal_confidence=decision['confidence'],
            account_balance=self.portfolio.balance,
            entry_price=decision['entry_price'],
            stop_loss=decision['stop_loss'],
            volatility=decision['volatility'],
            historical_performance=self.portfolio.get_performance_stats()
        )
        
        # Execute order
        result = await self.execution.execute_order({
            'symbol': decision['symbol'],
            'side': decision['side'],
            'size': size,
            'entry_price': decision['entry_price'],
            'stop_loss': decision['stop_loss'],
            'take_profit': decision.get('take_profit')
        })
        
        logging.info(f"Order executed: {result}")
    
    def stop(self):
        """Gracefully stop orchestrator"""
        logging.info("Stopping orchestrator...")
        self.running = False

# Entry point
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load configuration
    config = load_config('config/')
    
    # Create orchestrator
    orchestrator = TradingOrchestrator(config)
    
    # Run
    try:
        asyncio.run(orchestrator.main_loop())
    except KeyboardInterrupt:
        orchestrator.stop()
        logging.info("Shutdown complete")
```

---

## Deployment Commands

### Setup

```bash
# 1. Clone repository and install dependencies
conda env create -f environment.yml
conda activate crypto-trading

# 2. Setup database
psql -U postgres -c "CREATE DATABASE crypto_trading;"
psql -U postgres -d crypto_trading -f sql/schema.sql

# 3. Configure API keys
cp config/exchanges.yaml.example config/exchanges.yaml
# Edit config/exchanges.yaml with your API keys

# 4. Start Redis (for caching)
redis-server --daemonize yes

# 5. Fetch historical data (initial backfill)
python scripts/backfill_data.py --symbols BTC/USDT ETH/USDT --start 2023-01-01

# 6. Engineer features
python scripts/generate_features.py --symbols BTC/USDT ETH/USDT

# 7. Train models
python scripts/train_lightgbm.py --symbol BTC/USDT
python scripts/train_lstm.py --symbol BTC/USDT
python scripts/train_gnn.py --symbols BTC/USDT ETH/USDT SOL/USDT

# 8. Train RL agents
python agents/training/train_strategic.py
python agents/training/train_tactical.py
python agents/training/train_execution.py

# 9. Backtest system
python scripts/backtest.py --start 2024-01-01 --end 2024-12-01

# 10. Paper trade (dry run with real data, no real trades)
python orchestrator.py --mode paper --duration 7d

# 11. Live trading (START SMALL!)
python orchestrator.py --mode live --capital 1000
```

### Monitoring

```bash
# View logs
tail -f trading.log

# Monitor performance metrics
python monitoring/dashboard.py  # Opens Streamlit dashboard at http://localhost:8501

# Check database
psql -U postgres -d crypto_trading -c "SELECT * FROM portfolio_state ORDER BY time DESC LIMIT 10;"

# View TensorBoard (RL training)
tensorboard --logdir ./logs/
```

### Maintenance

```bash
# Daily: Retrain models with new data
python scripts/retrain_daily.py

# Weekly: Full retraining
python scripts/retrain_full.py

# Backup database
pg_dump -U postgres crypto_trading > backups/crypto_trading_$(date +%Y%m%d).sql
```

---

## Success Metrics & Validation

### Before Going Live:

1. **Backtest Performance** (minimum thresholds):
   - Sharpe Ratio > 1.5
   - Max Drawdown < 15%
   - Win Rate > 45%
   - Positive returns across multiple regimes

2. **Walk-Forward Validation**:
   - Out-of-sample testing on unseen data
   - Performance stable across time periods

3. **Paper Trading** (minimum 2 weeks):
   - Execution working correctly
   - Risk limits enforced
   - No critical bugs

4. **Start Small**:
   - Begin with $1000-2000
   - Scale up only after 1+ month profitable

### Ongoing Monitoring:

- Daily: Check for anomalies, review trades
- Weekly: Retrain models, update parameters
- Monthly: Full performance review, strategy adjustment

---

## Final Notes

**This specification provides everything needed for Opus 4.5 to implement the system.** Each module has:
- Clear purpose and responsibilities
- Concrete code examples
- Integration points with other modules
- Configuration examples

**Implementation priority:**
1. Start with data infrastructure (Week 1)
2. Get LightGBM baseline working (Week 2)
3. Add on-chain and sentiment data (Week 2)
4. Implement ensemble (Week 3)
5. Add RL agents (Week 4-5)
6. Risk management and execution (Week 5-6)
7. Paper trade extensively before live deployment

**Remember:** The research shows on-chain analytics and filtered sentiment provide the strongest validated signals. Don't skip proper backtesting and paper trading. Start with small capital and scale gradually.

Good luck! 🚀
