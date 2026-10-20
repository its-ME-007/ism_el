"""Pydantic schemas for the tiering system."""

from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import numpy as np


class Category(str, Enum):
    """Enterprise file categories."""
    FINANCE = "Finance"
    LEGAL = "Legal"
    HR = "HR"
    ENGINEERING = "Engineering"
    RESEARCH = "Research"
    MARKETING = "Marketing"
    OPERATIONS = "Operations"
    SECURITY = "Security"
    SYSTEM = "System"


class OperationType(str, Enum):
    """File operation types."""
    READ = "read"
    WRITE = "write"


class Tier(str, Enum):
    """Storage tier classification."""
    HOT = "HOT"
    WARM = "WARM"
    COLD = "COLD"


class WorkloadPatternType(str, Enum):
    """Workload pattern types."""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    SEASONAL = "seasonal"


class FileDocument(BaseModel):
    """Represents a file in the synthetic dataset."""
    file_id: str = Field(..., description="Unique identifier")
    filename: str = Field(..., description="Generated filename")
    category: Category = Field(..., description="Enterprise category")
    content: str = Field(..., description="Text content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    size_bytes: int = Field(..., ge=0, description="File size")
    created_at: datetime = Field(..., description="Creation timestamp")
    modified_at: datetime = Field(..., description="Last modification timestamp")
    
    model_config = ConfigDict(use_enum_values=True)


class AccessEvent(BaseModel):
    """Represents a file access event in the trace."""
    timestamp: datetime = Field(..., description="Event timestamp with nanosecond precision")
    file_id: str = Field(..., description="Referenced file")
    operation: OperationType = Field(..., description="Access type")
    size_bytes: Optional[int] = Field(None, description="Bytes transferred")
    
    model_config = ConfigDict(use_enum_values=True)


class WorkloadPattern(BaseModel):
    """Workload pattern specification for a file."""
    file_id: str
    pattern_type: WorkloadPatternType
    frequency: Optional[float] = Field(None, ge=0, description="Access frequency in accesses per hour")
    seasonal_period_hours: Optional[int] = Field(None, gt=0)
    seasonal_amplitude: Optional[float] = Field(None, ge=0.0, le=10.0)
    
    model_config = ConfigDict(use_enum_values=True)


class WorkloadPatternSpec(BaseModel):
    """Workload pattern specification (alias for WorkloadPattern)."""
    file_id: str
    pattern_type: WorkloadPatternType
    frequency: Optional[float] = Field(None, ge=0, description="Access frequency in accesses per hour")
    seasonal_period_hours: Optional[int] = Field(None, gt=0)
    seasonal_amplitude: Optional[float] = Field(None, ge=0.0, le=10.0)
    
    model_config = ConfigDict(use_enum_values=True)


class TierConfig(BaseModel):
    """Configuration for a storage tier."""
    tier: Tier
    latency_ms: float = Field(..., ge=0, description="Average access latency in milliseconds")
    cost_per_gb_month: float = Field(..., ge=0, description="Storage cost per GB per month")
    capacity_gb: float = Field(..., gt=0, description="Maximum capacity in GB")
    throughput_mbps: float = Field(..., ge=0, description="Throughput in MB/s")
    
    model_config = ConfigDict(use_enum_values=True)


class WorkloadFeatures(BaseModel):
    """Time-series features from access patterns."""
    file_id: str
    
    # Access counts
    access_count_1h: int = Field(ge=0)
    access_count_24h: int = Field(ge=0)
    access_count_7d: int = Field(ge=0)
    access_count_30d: int = Field(ge=0)
    
    # Recency metrics
    last_access_hours: float = Field(ge=0)
    time_since_creation_hours: float = Field(ge=0)
    
    # Inter-arrival statistics
    mean_inter_arrival_hours: float = Field(ge=0)
    std_inter_arrival_hours: float = Field(ge=0)
    
    # Pattern indicators
    read_write_ratio: float = Field(ge=0)
    periodicity_score: float = Field(ge=0, le=1)


class SemanticFeatures(BaseModel):
    """Semantic features from content analysis."""
    file_id: str
    tfidf_vector: Optional[List[float]] = None
    embedding_vector: Optional[List[float]] = None
    category_confidence: float = Field(ge=0, le=1)


class SeasonalityResult(BaseModel):
    """Result from seasonality detection."""
    file_id: str
    detected: bool
    period_hours: Optional[int] = None
    amplitude: Optional[float] = None
    phase: Optional[float] = None
    confidence: float = Field(ge=0, le=1)


class RAGContext(BaseModel):
    """Contextual features from RAG extraction."""
    file_id: str
    category_confidence: float = Field(ge=0.0, le=1.0)
    related_files: List[str] = Field(default_factory=list)
    context_summary: str = Field(default="")
    feature_vector: List[float] = Field(default_factory=list)


class EmbeddingRecord(BaseModel):
    """Single embedding entry in the knowledge base."""
    file_id: str
    embedding: List[float]
    model_version: str
    created_at: datetime


class TierPrediction(BaseModel):
    """Tier prediction result."""
    file_id: str
    predicted_tier: Tier
    probabilities: Dict[str, float]  # Tier -> probability
    confidence: float = Field(ge=0, le=1)
    
    model_config = ConfigDict(use_enum_values=True)


class FutureAccessPrediction(BaseModel):
    """Future access probability prediction."""
    file_id: str
    access_probability: float = Field(ge=0.0, le=1.0)
    prediction_window_hours: int = Field(gt=0)
    confidence: float = Field(ge=0.0, le=1.0)


class MigrationDecision(BaseModel):
    """Migration decision for a file."""
    file_id: str
    current_tier: Tier
    recommended_tier: Tier
    confidence: float = Field(ge=0, le=1)
    rationale: str
    
    model_config = ConfigDict(use_enum_values=True)


class MigrationCost(BaseModel):
    """Cost model for file migration."""
    file_id: str
    source_tier: Tier
    dest_tier: Tier
    latency_overhead_ms: float = Field(ge=0)
    bandwidth_cost: float = Field(ge=0)
    total_cost: float = Field(ge=0)
    
    model_config = ConfigDict(use_enum_values=True)


class DatasetConfig(BaseModel):
    """Configuration for dataset generation."""
    num_files: int = Field(gt=0)
    seed: Optional[int] = None
    category_distribution: Optional[Dict[str, float]] = None
    size_mean_mb: float = Field(gt=0, default=100.0)
    size_std_mb: float = Field(ge=0, default=50.0)
    
    model_config = ConfigDict(use_enum_values=True)


class WorkloadConfig(BaseModel):
    """Configuration for workload generation."""
    duration_hours: float = Field(gt=0)
    seed: Optional[int] = None
    hot_ratio: float = Field(ge=0, le=1, default=0.2)
    warm_ratio: float = Field(ge=0, le=1, default=0.3)
    cold_ratio: float = Field(ge=0, le=1, default=0.4)
    seasonal_ratio: float = Field(ge=0, le=1, default=0.1)
    
    model_config = ConfigDict(use_enum_values=True)


class SimulationConfig(BaseModel):
    """Configuration for simulation."""
    duration_hours: float = Field(gt=0)
    decision_interval_hours: float = Field(gt=0, default=1.0)
    seed: Optional[int] = None
    
    model_config = ConfigDict(use_enum_values=True)


class SimulationMetrics(BaseModel):
    """Metrics collected during simulation."""
    total_accesses: int = Field(ge=0)
    total_migrations: int = Field(ge=0)
    avg_access_latency_ms: float = Field(ge=0)
    p95_access_latency_ms: float = Field(ge=0)
    total_storage_cost: float = Field(ge=0)
    cache_hit_rate: float = Field(ge=0, le=1)
    

class SimulationResult(BaseModel):
    """Complete result from a simulation run."""
    metrics: SimulationMetrics
    migrations: List[MigrationDecision]
    tier_distribution_timeline: List[Dict[str, float]]
    execution_time_seconds: float = Field(ge=0)
