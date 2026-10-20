"""XGBoost-based tier classification model."""

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import xgboost
from sklearn.preprocessing import StandardScaler
from src.tiering.ingestion.schemas import Tier


class TierClassifier:
    """
    XGBoost multiclass classifier for HOT(0)/WARM(1)/COLD(2) tier prediction.
    
    Predicts optimal storage tier based on workload, semantic, and/or future-access features.
    Outputs probability distributions over tier classes.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize TierClassifier with XGBoost parameters.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate for boosting
            random_state: Random seed for reproducibility
        """
        self.model = xgboost.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=random_state,
            verbosity=0
        )
        self.feature_names: Optional[List[str]] = None
        self.scaler: Optional[StandardScaler] = None
        self._is_fitted = False
        self.model_version = "1.0"
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> None:
        """
        Train XGBClassifier on training data.
        
        Args:
            X: Training feature matrix (n_samples, n_features)
            y: Target labels (0=HOT, 1=WARM, 2=COLD)
            feature_names: Optional names for features (for interpretation)
            eval_set: Optional (X_val, y_val) for early stopping
            
        **Validates: Requirements 9.1, 9.2**
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Scale features for XGBoost
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        fit_kwargs = {"verbose": False}
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_scaled = self.scaler.transform(X_val)
            fit_kwargs['eval_set'] = [(X_val_scaled, y_val)]
            # Note: early stopping is not available in XGBClassifier.fit() in current version
        
        self.model.fit(X_scaled, y, **fit_kwargs)
        self._is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict tier class for samples.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted class indices (0=HOT, 1=WARM, 2=COLD)
            
        **Validates: Requirements 9.3**
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability distribution over tier classes.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Probability matrix (n_samples, 3) where each row sums to 1
            Column indices: 0=HOT, 1=WARM, 2=COLD
            
        **Validates: Requirements 9.2**
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> Dict[str, float]:
        """
        Extract feature importance scores from model.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover', 'total_gain')
            
        Returns:
            Dictionary mapping feature names to normalized importance scores [0, 1]
            
        **Validates: Requirements 9.5**
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before extracting importance")
        
        importances = self.model.get_booster().get_score(importance_type=importance_type)
        feature_imp = {}
        
        for feat_idx_str, score in importances.items():
            try:
                # XGBoost returns feature indices as "f0", "f1", etc.
                idx = int(feat_idx_str[1:])
                if idx < len(self.feature_names):
                    feature_imp[self.feature_names[idx]] = score
            except (ValueError, IndexError):
                pass
        
        # Normalize to [0, 1]
        if feature_imp:
            max_val = max(feature_imp.values())
            if max_val > 0:
                feature_imp = {k: v / max_val for k, v in feature_imp.items()}
        
        return feature_imp
    
    def save(self, path: str, version: Optional[str] = None) -> None:
        """
        Save trained model to disk.
        
        Args:
            path: Base path for model file (extension and version appended)
            version: Model version string (default: "1.0")
            
        **Validates: Requirements 9.5**
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        version = version or self.model_version
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        model_file = path_obj.parent / f"{path_obj.stem}_xgb_{version}.json"
        self.model.get_booster().save_model(str(model_file))
        
        # Also save the scaler using joblib
        import joblib
        scaler_file = path_obj.parent / f"{path_obj.stem}_scaler_{version}.pkl"
        joblib.dump(self.scaler, str(scaler_file))
    
    @classmethod
    def load(cls, path: str, version: Optional[str] = None) -> 'TierClassifier':
        """
        Load trained model from disk.
        
        Args:
            path: Base path for model file
            version: Model version string (default: "1.0")
            
        Returns:
            Loaded TierClassifier instance
            
        **Validates: Requirements 9.5**
        """
        version = version or "1.0"
        path_obj = Path(path)
        model_file = path_obj.parent / f"{path_obj.stem}_xgb_{version}.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Create new instance and load model
        classifier = cls()
        classifier.model = xgboost.XGBClassifier()
        classifier.model.load_model(str(model_file))
        classifier._is_fitted = True
        classifier.model_version = version
        
        # Load the scaler
        import joblib
        scaler_file = path_obj.parent / f"{path_obj.stem}_scaler_{version}.pkl"
        if scaler_file.exists():
            classifier.scaler = joblib.load(str(scaler_file))
        else:
            classifier.scaler = StandardScaler()
        
        return classifier
    
    def get_classification_report(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Generate classification report with precision, recall, F1 per class.
        
        Args:
            X: Test feature matrix
            y: Ground truth labels
            
        Returns:
            Dictionary with classification metrics
            
        **Validates: Requirements 9.6**
        """
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            accuracy_score, confusion_matrix
        )
        
        y_pred = self.predict(X)
        
        return {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision_weighted': float(precision_score(y, y_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y, y_pred, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(y, y_pred, average='weighted', zero_division=0)),
            'precision_hot': float(precision_score(y, y_pred, labels=[0], average='weighted', zero_division=0)),
            'recall_hot': float(recall_score(y, y_pred, labels=[0], average='weighted', zero_division=0)),
            'f1_hot': float(f1_score(y, y_pred, labels=[0], average='weighted', zero_division=0)),
            'precision_warm': float(precision_score(y, y_pred, labels=[1], average='weighted', zero_division=0)),
            'recall_warm': float(recall_score(y, y_pred, labels=[1], average='weighted', zero_division=0)),
            'f1_warm': float(f1_score(y, y_pred, labels=[1], average='weighted', zero_division=0)),
            'precision_cold': float(precision_score(y, y_pred, labels=[2], average='weighted', zero_division=0)),
            'recall_cold': float(recall_score(y, y_pred, labels=[2], average='weighted', zero_division=0)),
            'f1_cold': float(f1_score(y, y_pred, labels=[2], average='weighted', zero_division=0)),
        }
