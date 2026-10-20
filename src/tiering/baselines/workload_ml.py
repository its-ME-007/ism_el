"""Workload-ML baseline: uses only workload features for tier prediction."""

from typing import List, Dict
import numpy as np
from src.tiering.models.tier_classifier import TierClassifier
from src.tiering.ingestion.schemas import Tier


class WorkloadMLBaseline:
    """
    ML-based baseline using ONLY workload features for tier prediction.
    
    This baseline trains an XGBoost classifier on workload features alone
    (no semantic features, no future-access predictions). It serves as
    the comparison point to measure the value of semantic and predictive features.
    
    **Validates: Requirements 15.1, 15.2, 15.3, 15.4**
    """
    
    def __init__(
        self,
        classifier: TierClassifier,
        workload_feature_indices: List[int]
    ):
        """
        Initialize Workload-ML baseline.
        
        Args:
            classifier: Trained TierClassifier instance
            workload_feature_indices: Indices of workload features in full feature vector
                                     (e.g., [0, 1, 2, 3, 4] for access counts and recency)
        """
        self.classifier = classifier
        self.workload_indices = workload_feature_indices
    
    def predict_tier(self, features: np.ndarray) -> np.ndarray:
        """
        Predict tier using only workload features.
        
        Args:
            features: Full feature matrix (n_samples, n_total_features)
            
        Returns:
            Predicted tier for each sample (n_samples,) with values from Tier enum
            
        **Validates: Requirements 15.1, 15.2, 15.4**
        """
        # Extract only workload features
        workload_features = features[:, self.workload_indices]
        
        # Get class predictions
        class_indices = self.classifier.predict(workload_features)
        
        # Map class indices to tier enums
        tier_map = {0: Tier.HOT, 1: Tier.WARM, 2: Tier.COLD}
        tiers = np.array([tier_map[idx] for idx in class_indices], dtype=object)
        
        return tiers
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get probability distribution over tiers using only workload features.
        
        Args:
            features: Full feature matrix (n_samples, n_total_features)
            
        Returns:
            Probability matrix (n_samples, 3) where columns are [HOT, WARM, COLD]
            
        **Validates: Requirements 15.4**
        """
        # Extract only workload features
        workload_features = features[:, self.workload_indices]
        
        # Get probability predictions
        return self.classifier.predict_proba(workload_features)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance rankings from classifier.
        
        Returns:
            Dictionary mapping feature names to normalized importance scores
            
        **Validates: Requirements 15.3**
        """
        return self.classifier.get_feature_importance()
    
    @classmethod
    def train(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        workload_feature_indices: List[int],
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> 'WorkloadMLBaseline':
        """
        Train a new Workload-ML baseline from scratch.
        
        Args:
            X_train: Training feature matrix (n_train, n_features)
            y_train: Training labels (0=HOT, 1=WARM, 2=COLD)
            workload_feature_indices: Indices of workload features
            X_val: Optional validation feature matrix
            y_val: Optional validation labels
            
        Returns:
            Trained WorkloadMLBaseline instance
            
        **Validates: Requirements 15.1, 15.2**
        """
        # Extract workload features only
        X_train_workload = X_train[:, workload_feature_indices]
        
        # Train classifier
        classifier = TierClassifier()
        feature_names = [f"workload_feat_{i}" for i in range(len(workload_feature_indices))]
        
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_workload = X_val[:, workload_feature_indices]
            eval_set = (X_val_workload, y_val)
        
        classifier.fit(X_train_workload, y_train, feature_names=feature_names, eval_set=eval_set)
        
        return cls(classifier, workload_feature_indices)
