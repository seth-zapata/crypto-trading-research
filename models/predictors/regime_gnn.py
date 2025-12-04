"""
GNN Regime Detector
===================

Graph Neural Network for detecting market regime shifts.

Architecture:
- Nodes: Crypto assets (BTC, ETH, SOL, etc.)
- Edges: Fully connected (learns attention weights)
- Node features: Returns, volatility, momentum for each asset
- Output: 3-class classification (RISK_ON, CAUTION, RISK_OFF)

Goal: Detect regime shifts 3-5 days BEFORE major drawdowns.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool

logger = logging.getLogger(__name__)


class RegimeGNN(nn.Module):
    """
    Graph Attention Network for regime detection.

    Uses attention mechanism to learn which cross-asset relationships
    are most informative for regime prediction.
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        """
        Initialize GNN.

        Args:
            num_node_features: Number of features per asset node
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            num_classes: Number of regime classes (3)
            dropout: Dropout rate
        """
        super().__init__()

        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim

        # Graph attention layers
        self.conv1 = GATConv(
            num_node_features,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )

        self.conv2 = GATConv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=False
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyG Data object with x, edge_index, batch

        Returns:
            Logits for each regime class
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph attention layers
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # Global pooling (aggregate all node features)
        x = global_mean_pool(x, batch)

        # Classification
        logits = self.classifier(x)

        return logits


class RegimeDetector:
    """
    High-level interface for regime detection.

    Handles data preparation, training, and inference.
    """

    REGIME_LABELS = ['RISK_ON', 'CAUTION', 'RISK_OFF']
    REGIME_TO_IDX = {r: i for i, r in enumerate(REGIME_LABELS)}

    def __init__(
        self,
        assets: List[str] = ['BTC', 'ETH', 'SOL'],
        lookback: int = 20,
        hidden_dim: int = 64,
        device: str = 'auto'
    ):
        """
        Initialize detector.

        Args:
            assets: List of asset symbols
            lookback: Lookback window for features
            hidden_dim: GNN hidden dimension
            device: 'cuda', 'cpu', or 'auto'
        """
        self.assets = assets
        self.lookback = lookback
        self.hidden_dim = hidden_dim

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.feature_cols = None

    def _prepare_node_features(self, df: pd.DataFrame, idx: int) -> torch.Tensor:
        """
        Prepare node features for a single timestep.

        Each asset becomes a node with its own feature vector.
        """
        features = []

        for asset in self.assets:
            asset_features = []

            # Returns at different horizons
            for period in [1, 5, 20]:
                col = f'{asset}_return_{period}d'
                if col in df.columns:
                    asset_features.append(df[col].iloc[idx])

            # Volatility
            for window in [10, 20, 60]:
                col = f'{asset}_vol_{window}d'
                if col in df.columns:
                    asset_features.append(df[col].iloc[idx])

            # Correlation to BTC (if not BTC)
            if asset != 'BTC':
                col = f'{asset}_corr_to_BTC'
                if col in df.columns:
                    asset_features.append(df[col].iloc[idx])
            else:
                asset_features.append(1.0)  # BTC correlation to itself

            features.append(asset_features)

        # Pad features to same length
        max_len = max(len(f) for f in features)
        features = [f + [0.0] * (max_len - len(f)) for f in features]

        return torch.tensor(features, dtype=torch.float32)

    def _create_graph(self, node_features: torch.Tensor) -> Data:
        """
        Create a fully connected graph from node features.
        """
        num_nodes = node_features.shape[0]

        # Fully connected edges (including self-loops)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=node_features, edge_index=edge_index)

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        labels: pd.Series
    ) -> Tuple[List[Data], List[int]]:
        """
        Prepare graph dataset from features and labels.

        Returns:
            Tuple of (list of graphs, list of label indices)
        """
        graphs = []
        targets = []

        valid_idx = df.dropna().index.intersection(labels.dropna().index)

        for idx in range(len(valid_idx)):
            try:
                node_features = self._prepare_node_features(df.loc[valid_idx], idx)

                # Handle NaN
                if torch.isnan(node_features).any():
                    continue

                graph = self._create_graph(node_features)
                label = self.REGIME_TO_IDX[labels.loc[valid_idx].iloc[idx]]

                graphs.append(graph)
                targets.append(label)

            except Exception as e:
                logger.warning(f"Error preparing sample {idx}: {e}")
                continue

        # Store number of features for model initialization
        if graphs:
            self.num_node_features = graphs[0].x.shape[1]

        return graphs, targets

    def train(
        self,
        train_graphs: List[Data],
        train_labels: List[int],
        val_graphs: Optional[List[Data]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
        class_weights: Optional[List[float]] = None
    ) -> Dict:
        """
        Train the GNN model.

        Args:
            train_graphs: Training graphs
            train_labels: Training labels
            val_graphs: Validation graphs
            val_labels: Validation labels
            epochs: Number of epochs
            lr: Learning rate
            batch_size: Batch size
            class_weights: Optional class weights for imbalanced data

        Returns:
            Training history dict
        """
        # Initialize model
        if self.model is None:
            self.model = RegimeGNN(
                num_node_features=self.num_node_features,
                hidden_dim=self.hidden_dim,
                num_classes=len(self.REGIME_LABELS)
            ).to(self.device)

        # Class weights for imbalanced classes
        if class_weights is None:
            # Calculate from training data
            counts = np.bincount(train_labels, minlength=3)
            weights = 1.0 / (counts + 1)
            weights = weights / weights.sum() * len(weights)
            class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0

            # Shuffle and batch
            indices = np.random.permutation(len(train_graphs))

            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_graphs = [train_graphs[j] for j in batch_idx]
                batch_labels = [train_labels[j] for j in batch_idx]

                batch = Batch.from_data_list(batch_graphs).to(self.device)
                labels_tensor = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch)
                loss = criterion(logits, labels_tensor)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_idx)
                train_correct += (logits.argmax(dim=1) == labels_tensor).sum().item()

            train_loss /= len(train_graphs)
            train_acc = train_correct / len(train_graphs)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Validation
            if val_graphs is not None:
                val_loss, val_acc = self._evaluate(val_graphs, val_labels, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                scheduler.step(val_loss)

                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                        f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
                    )
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}")

        return history

    def _evaluate(
        self,
        graphs: List[Data],
        labels: List[int],
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Evaluate model on a dataset."""
        self.model.eval()

        with torch.no_grad():
            batch = Batch.from_data_list(graphs).to(self.device)
            labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)

            logits = self.model(batch)
            loss = criterion(logits, labels_tensor).item()
            acc = (logits.argmax(dim=1) == labels_tensor).float().mean().item()

        return loss, acc

    def predict(self, graphs: List[Data]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regimes for graphs.

        Returns:
            Tuple of (predicted labels, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()

        with torch.no_grad():
            batch = Batch.from_data_list(graphs).to(self.device)
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

        return preds, probs

    def predict_regime(self, df: pd.DataFrame, idx: int = -1) -> Dict:
        """
        Predict regime for a specific timestep.

        Args:
            df: Feature DataFrame
            idx: Index to predict (-1 for latest)

        Returns:
            Dict with regime, confidence, probabilities
        """
        if idx == -1:
            idx = len(df) - 1

        node_features = self._prepare_node_features(df, idx)
        graph = self._create_graph(node_features)

        preds, probs = self.predict([graph])

        regime = self.REGIME_LABELS[preds[0]]
        confidence = probs[0].max()

        return {
            'regime': regime,
            'confidence': confidence,
            'probabilities': {
                r: probs[0][i] for i, r in enumerate(self.REGIME_LABELS)
            }
        }

    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state': self.model.state_dict(),
            'assets': self.assets,
            'lookback': self.lookback,
            'hidden_dim': self.hidden_dim,
            'num_node_features': self.num_node_features,
        }, path)

    def load(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)

        self.assets = checkpoint['assets']
        self.lookback = checkpoint['lookback']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_node_features = checkpoint['num_node_features']

        self.model = RegimeGNN(
            num_node_features=self.num_node_features,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.REGIME_LABELS)
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state'])


if __name__ == "__main__":
    # Test the module
    import sys
    sys.path.insert(0, '.')

    logging.basicConfig(level=logging.INFO)

    from data.ingestion.multi_asset import build_regime_detection_features

    print("Building features...")
    df, labels = build_regime_detection_features(
        start_date='2020-01-01',
        assets=['BTC', 'ETH', 'SOL']
    )

    print(f"Dataset: {len(df)} samples")
    print(f"Regime distribution: {labels.value_counts().to_dict()}")

    # Initialize detector
    detector = RegimeDetector(assets=['BTC', 'ETH', 'SOL'])

    # Prepare graphs
    print("\nPreparing graphs...")
    graphs, targets = detector.prepare_dataset(df, labels)
    print(f"Prepared {len(graphs)} graphs")

    # Train/val split (time series)
    split = int(len(graphs) * 0.8)
    train_graphs, val_graphs = graphs[:split], graphs[split:]
    train_labels, val_labels = targets[:split], targets[split:]

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    # Train
    print("\nTraining GNN...")
    history = detector.train(
        train_graphs, train_labels,
        val_graphs, val_labels,
        epochs=50,
        batch_size=32
    )

    # Final evaluation
    preds, probs = detector.predict(val_graphs)
    from sklearn.metrics import classification_report
    print("\nValidation Results:")
    print(classification_report(
        val_labels, preds,
        target_names=detector.REGIME_LABELS
    ))
