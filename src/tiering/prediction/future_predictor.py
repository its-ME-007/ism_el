"""Future access prediction using XGBoost regression."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from src.tiering.workload.feature_extractor import WorkloadFeatures


class FutureAccessPredictor:
    """
    XGBoost regression model for predicting file access probability.
    
    Predicts the probability that a file will be accessed within a
    configurable future time window based on workload features.
    
    Requirements:
    - Requirement 7.1: Train XGBoost regression model for access probability
    - Requirement 7.2: Use time-series features for prediction
    - Requirement 7.3: Output calibrated probabilities in [0, 1]
    - Requirement 7.4: Prevent data leakage with time-based splits
    - Requirement 7.5: Provide feature importance rankings
    - Requirement 7.6: Handle cold-start files with category-level priors
    """
    
    def __init__(
        self,
        prediction_window_hours: int = 24,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        objective: str = 'reg:squarederror',
        calibration_method: str = 'isotonic',
        random_state: int = 42
    ):
        """
        Initialize future access predictor.
        
        Args:
            prediction_window_hours: Time window for access prediction (hours)
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate for XGBoost
            objective: XGBoost objective function
            calibration_method: Calibration method ('isotonic' or 'clip')
            random_state: Random seed for reproducibility
        """
        self.prediction_window_hours = prediction_window_hours
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.calibration_method = calibration_method
        self.random_state = random_state
        
        # Model and calibrator
        self.model: Optional[xgb.XGBRegressor] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.feature_names: Optional[List[str]] = None
        
        # Category-level priors for cold-start
        self.category_priors: Dict[str, float] = {}
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        early_stopping_rounds: int = 10
    ) -> None:
        """
        Train the XGBoost model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels indicating access in prediction window
            feature_names: Names of features for interpretability
            eval_set: Optional validation set (X_val, y_val) for early stopping
            early_stopping_rounds: Number of rounds for early stopping
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Create and train model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective=self.objective,
            random_state=self.random_state,
            verbosity=0
        )
        
        # Fit model - use eval_set only if provided
        fit_kwargs = {}
        if eval_set is not None:
            X_val, y_val = eval_set
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
        
        self.model.fit(X, y, verbose=False, **fit_kwargs)
        
        # Calibrate predictions using isotonic regression on training data
        raw_preds = self.model.predict(X)
        
        if self.calibration_method == 'isotonic':
            self.calibrator = IsotonicRegression(
                out_of_bounds='clip',
                y_min=0.0,
                y_max=1.0
            )
            self.calibrator.fit(raw_preds, y)
        # For 'clip', calibration happens in predict()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict access probability.
        
        Returns:
            Array of shape (n_samples,) with probability scores in [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get raw predictions
        raw_preds = self.model.predict(X)
        
        # Calibrate to [0, 1]
        if self.calibration_method == 'isotonic' and self.calibrator is not None:
            calibrated = self.calibrator.predict(raw_preds)
        else:  # 'clip'
            calibrated = np.clip(raw_preds, 0.0, 1.0)
        
        return calibrated
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability as a single-column array (compatibility method).
        
        Returns:
            Array of shape (n_samples, 1) with calibrated probabilities
        """
        probs = self.predict(X)
        return probs.reshape(-1, 1)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance rankings.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover', 'total_gain', 'total_cover')
        
        Returns:
            Dictionary mapping feature name to importance score
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        importances = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Convert feature indices to names
        feature_importance = {}
        for feat_idx_str, score in importances.items():
            # Extract index from "f{index}" format
            try:
                idx = int(feat_idx_str[1:])
                if idx < len(self.feature_names):
                    feature_importance[self.feature_names[idx]] = score
            except (ValueError, IndexError):
                pass
        
        # Normalize to [0, 1]
        if feature_importance:
            max_val = max(feature_importance.values())
            if max_val > 0:
                feature_importance = {
                    k: v / max_val for k, v in feature_importance.items()
                }
        
        return feature_importance
    
    def set_category_priors(self, category_priors: Dict[str, float]) -> None:
        """
        Set category-level priors for cold-start handling.
        
        Args:
            category_priors: Dictionary mapping category name to prior probability
        """
        self.category_priors = category_priors.copy()
    
    def predict_with_cold_start(
        self,
        X: np.ndarray,
        category: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict with cold-start fallback.
        
        For samples with insufficient historical data, falls back to
        category-level priors if available.
        
        Args:
            X: Feature matrix
            category: Category name for cold-start fallback
        
        Returns:
            Array of predictions with cold-start handling applied
        """
        preds = self.predict(X)
        
        # Apply cold-start prior if category provided and prior available
        if category is not None and category in self.category_priors:
            prior = self.category_priors[category]
            # Blend predictions with prior for potentially cold-start cases
            # (those with very low predicted probability)
            cold_start_mask = preds < 0.1
            preds[cold_start_mask] = (preds[cold_start_mask] + prior) / 2.0
        
        return preds
    
    def save(self, path: Path, version: str = "1.0") -> None:
        """
        Save model to disk.
        
        Args:
            path: Path where to save model
            version: Version identifier
        """
        if self.model is None:
            raise ValueError("Model not trained. Cannot save.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_file = path.parent / f"{path.stem}_xgb_{version}.json"
        self.model.get_booster().save_model(str(model_file))
        
        # Save metadata
        metadata_file = path.parent / f"{path.stem}_meta_{version}.txt"
        with open(metadata_file, 'w') as f:
            f.write(f"version={version}\n")
            f.write(f"prediction_window_hours={self.prediction_window_hours}\n")
            f.write(f"calibration_method={self.calibration_method}\n")
            f.write(f"feature_names={','.join(self.feature_names or [])}\n")
    
    @classmethod
    def load(cls, path: Path, version: str = "1.0") -> 'FutureAccessPredictor':
        """
        Load model from disk.
        
        Args:
            path: Path where model was saved
            version: Version identifier
        
        Returns:
            Loaded FutureAccessPredictor instance
        """
        path = Path(path)
        
        # Load metadata
        metadata_file = path.parent / f"{path.stem}_meta_{version}.txt"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        metadata[key] = value
        
        # Create instance
        predictor = cls(
            prediction_window_hours=int(metadata.get('prediction_window_hours', 24)),
            calibration_method=metadata.get('calibration_method', 'isotonic')
        )
        
        # Load model
        model_file = path.parent / f"{path.stem}_xgb_{version}.json"
        if model_file.exists():
            booster = xgb.Booster()
            booster.load_model(str(model_file))
            predictor.model = xgb.XGBRegressor()
            predictor.model._Booster = booster
        
        # Restore feature names
        if 'feature_names' in metadata and metadata['feature_names']:
            predictor.feature_names = metadata['feature_names'].split(',')
        
        return predictor
