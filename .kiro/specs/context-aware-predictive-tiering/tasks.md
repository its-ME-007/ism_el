# Implementation Plan: Context-Aware Predictive Storage Tiering

## Overview

This implementation plan follows a phase-based approach for building the Context-Aware Predictive Storage Tiering system. The system evaluates whether semantic and contextual information retrieved through RAG, combined with workload characteristics and predicted future access patterns, can improve storage-tier placement decisions.

**Implementation Language**: Python 3.10+

**Phase Structure**:
- **Phase 1 - MVP (Foundation)**: Synthetic data generation, workload traces, conventional baselines, SimPy simulator
- **Phase 2 - ML Layer**: Workload features, future access prediction, tier classifier, Workload-ML baseline
- **Phase 3 - Semantic Layer**: TF-IDF, embeddings, FAISS knowledge base, RAG context extraction
- **Phase 4 - Integration & Migration**: Feature fusion, migration controller, full context-aware classifier
- **Phase 5 - Experiments & Evaluation**: Experiment framework, automated runners, metrics, visualization, ablation studies

---

## Tasks

- [x] 1. Project Structure and Configuration System
  - **Phase:** 1
  - **Priority:** Critical
  - **Complexity:** Small
  - **Dependencies:** None
  - **Description:** Set up the complete project directory structure, configuration loading system, and core data models using Pydantic.
  - **Acceptance Criteria:**
    - Create directory structure as specified in design.md
    - Implement configuration loader with YAML support
    - Implement configuration validation against JSON schemas
    - Create core Pydantic data models: FileDocument, AccessEvent, Tier, TierConfig
    - Implement seed management for reproducibility
  - **Implementation Notes:** Create `src/tiering/__init__.py` with package initialization. Use `pydantic>=2.0.0` for all data models. Configuration files go in `configs/` directory.
  - _Requirements: 20.1, 20.6, 26.1, 26.2, 26.3, 26.4, 26.5, 26.6, 26.7_

  - [x]* 1.1 Write unit tests for configuration system
    - Test YAML loading with valid and invalid configs
    - Test schema validation errors
    - Test seed management reproducibility
    - _Requirements: 20.1, 20.6_

  - [x]* 1.2 Write property tests for Pydantic models
    - **Property 16: Configuration Validation**
    - **Validates: Requirements 20.1, 20.6**

- [x] 2. Synthetic Dataset Generation
  - **Phase:** 1
  - **Priority:** Critical
  - **Complexity:** Medium
  - **Dependencies:** Task 1
  - **Description:** Implement the DatasetGenerator component that generates synthetic files across nine enterprise categories with realistic content, sizes, and metadata.
  - **Acceptance Criteria:**
    - Generate files across all nine categories with configurable distribution
    - Assign unique identifiers, creation timestamps, sizes, and metadata
    - Generate realistic file names and content snippets per category
    - Produce reproducible output with identical seeds
    - Output to JSON Lines format with schema validation
  - **Implementation Notes:** Use templates for category-specific content. Implement `to_jsonl()` method for persistence. Store in `data/generated/` directory.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x]* 2.1 Write unit tests for dataset generation
    - Test category distribution
    - Test uniqueness of file IDs
    - Test JSON Lines output format
    - _Requirements: 1.1, 1.2, 1.6_

  - [x]* 2.2 Write property tests for dataset reproducibility
    - **Property 1: Dataset Generation Reproducibility**
    - **Validates: Requirements 1.3, 1.5**

  - [x]* 2.3 Write property tests for dataset schema compliance
    - **Property 2: Dataset Schema Compliance**
    - **Validates: Requirements 1.1, 1.2, 1.6**

- [x] 3. Workload Pattern Generation
  - **Phase:** 1
  - **Priority:** Critical
  - **Complexity:** Medium
  - **Dependencies:** Task 2
  - **Description:** Implement the WorkloadGenerator component that generates realistic access traces with hot, warm, cold, and seasonal patterns.
  - **Acceptance Criteria:**
    - Generate access traces with four pattern types (hot/warm/cold/seasonal)
    - Assign workload patterns based on configurable distribution
    - Generate events with nanosecond timestamp precision
    - Support time-based pattern evolution
    - Apply configurable periodicity and amplitude for seasonal patterns
  - **Implementation Notes:** Use numpy for efficient trace generation. Implement pattern evolution as a separate method. Output chronologically sorted events.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x]* 3.1 Write unit tests for workload generation
    - Test pattern assignment
    - Test chronological ordering
    - Test seasonal pattern generation
    - _Requirements: 2.1, 2.3, 2.5_

  - [x]* 3.2 Write property tests for trace chronology
    - **Property 3: Workload Trace Chronological Ordering**
    - **Validates: Requirements 2.3, 2.5**

- [x] 4. LRU Baseline Implementation
  - **Phase:** 1
  - **Priority:** High
  - **Complexity:** Small
  - **Dependencies:** Task 2
  - **Description:** Implement the LRU (Least Recently Used) baseline tiering policy.
  - **Acceptance Criteria:**
    - Track last access time for each file
    - Promote files to HOT tier on access, demote based on recency
    - Implement configurable tier capacity limits
    - No use of semantic or future-access information
  - **Implementation Notes:** Use OrderedDict or heapq for efficient LRU tracking. Implement as a class with `on_access()` and `on_decision_interval()` methods.
  - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [x]* 4.1 Write unit tests for LRU baseline
    - Test promotion on access
    - Test demotion when capacity exceeded
    - Test capacity enforcement
    - _Requirements: 12.1, 12.2, 12.3_

- [x] 5. LFU Baseline Implementation
  - **Phase:** 1
  - **Priority:** High
  - **Complexity:** Small
  - **Dependencies:** Task 2
  - **Description:** Implement the LFU (Least Frequently Used) baseline tiering policy.
  - **Acceptance Criteria:**
    - Track access frequency counts for each file
    - Promote frequently accessed files, demote rarely accessed
    - Implement configurable frequency counting windows
    - No use of semantic or future-access information
  - **Implementation Notes:** Use collections.Counter for frequency tracking. Support sliding window for frequency decay.
  - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [x]* 5.1 Write unit tests for LFU baseline
    - Test frequency counting
    - Test promotion/demotion logic
    - Test window-based frequency decay
    - _Requirements: 13.1, 13.2, 13.3_

- [x] 6. Age-Based Baseline Implementation
  - **Phase:** 1
  - **Priority:** High
  - **Complexity:** Small
  - **Dependencies:** Task 2
  - **Description:** Implement the age-based baseline tiering policy that migrates files based on creation/modification time.
  - **Acceptance Criteria:**
    - Track file creation and modification times
    - Migrate files to colder tiers based on age thresholds
    - Implement configurable age thresholds per tier transition
    - No use of semantic, workload, or future-access information
  - **Implementation Notes:** Simple threshold-based policy. Compare current time against file timestamps.
  - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [x]* 6.1 Write unit tests for age-based baseline
    - Test age threshold application
    - Test tier transition logic
    - Test edge cases (new files, very old files)
    - _Requirements: 14.1, 14.2, 14.3_

- [x] 7. SimPy Simulator Skeleton
  - **Phase:** 1
  - **Priority:** Critical
  - **Complexity:** Large
  - **Dependencies:** Task 3, Task 4, Task 5, Task 6
  - **Description:** Implement the core SimPy discrete event simulator framework for storage tier behavior.
  - **Acceptance Criteria:**
    - Model configurable storage tiers with latency, throughput, and cost
    - Process access events from workload traces
    - Route requests to appropriate tiers
    - Track system-level metrics (latency, cost, migration overhead)
    - Support reproducible runs with deterministic seeds
    - Output structured simulation logs
  - **Implementation Notes:** Use SimPy's Environment and Process concepts. Implement tier as a SimPy Resource. Create event generators from traces.
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

  - [x]* 7.1 Write unit tests for simulator core
    - Test event processing
    - Test tier routing
    - Test metric accumulation
    - _Requirements: 11.2, 11.4_

  - [x]* 7.2 Write property tests for simulation causality
    - **Property 14: Simulation Event Causality**
    - **Validates: Requirements 11.2, 21.4**

  - [x]* 7.3 Write property tests for simulation reproducibility
    - **Property 15: Simulation Reproducibility**
    - **Validates: Requirements 11.5, 20.3, 20.5**

- [x] 8. Phase 1 Checkpoint - MVP Validation
  - **Phase:** 1
  - **Priority:** Critical
  - **Complexity:** Small
  - **Dependencies:** Task 7
  - **Description:** Validate MVP phase completion with integration tests and baseline comparison.
  - **Acceptance Criteria:**
    - All unit and property tests pass
    - Integration test: Dataset → Traces → Simulation with each baseline
    - Produce comparable baseline metrics (LRU vs LFU vs Age)
    - Document baseline comparison results
  - **Implementation Notes:** Run simulation with each baseline on same dataset. Compare cost, latency, and migration metrics.
  - _Requirements: 25.1, 25.3_

- [x] 9. Workload Feature Extraction
  - **Phase:** 2
  - **Priority:** Critical
  - **Complexity:** Medium
  - **Dependencies:** Task 3
  - **Description:** Implement the WorkloadFeatureExtractor that computes time-series features from access traces.
  - **Acceptance Criteria:**
    - Compute access counts per configurable time windows (1h, 24h, 7d, 30d)
    - Compute recency metrics and inter-arrival statistics
    - Detect seasonal patterns using autocorrelation
    - Output normalized feature vectors for ML input
    - Prevent data leakage by using only historical data
  - **Implementation Notes:** Implement rolling window calculations. Use numpy for vectorized operations. Store in `src/tiering/workload/feature_extractor.py`.
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x]* 9.1 Write unit tests for workload feature extraction
    - Test access count calculations
    - Test recency metrics
    - Test empty history handling
    - _Requirements: 6.1, 6.2, 6.4_

  - [x]* 9.2 Write property tests for temporal integrity
    - **Property 7: Temporal Integrity in Feature Computation**
    - **Validates: Requirements 6.5, 7.4, 21.1, 21.2, 21.3**

  - [x]* 9.3 Write property tests for feature dimension consistency
    - **Property 4: Feature Vector Dimension Consistency**
    - **Validates: Requirements 6.4, 3.3, 3.5, 4.1, 4.2**

- [x] 10. Seasonality Detection
  - **Phase:** 2
  - **Priority:** Medium
  - **Complexity:** Medium
  - **Dependencies:** Task 9
  - **Description:** Implement the SeasonalityDetector for detecting periodic patterns in access traces.
  - **Acceptance Criteria:**
    - Detect seasonal patterns using autocorrelation analysis
    - Return period, amplitude, and phase when detected
    - Support configurable min/max period and confidence threshold
  - **Implementation Notes:** Use numpy.correlate for autocorrelation. Implement peak detection for period identification.
  - _Requirements: 2.6, 6.3_

  - [x]* 10.1 Write unit tests for seasonality detection
    - Test detection of known seasonal patterns
    - Test non-seasonal trace handling
    - _Requirements: 2.6, 6.3_

  - [x]* 10.2 Write property tests for seasonality detection accuracy
    - **Property 8: Seasonality Detection Accuracy**
    - **Validates: Requirements 2.6, 6.3**

- [x] 11. Future Access Predictor
  - **Phase:** 2
  - **Priority:** Critical
  - **Complexity:** Medium
  - **Dependencies:** Task 9
  - **Description:** Implement the FutureAccessPredictor XGBoost regression model for predicting access probability.
  - **Acceptance Criteria:**
    - Train XGBoost regression model for access probability prediction
    - Use time-series features from workload extractor
    - Output calibrated probability scores in [0, 1]
    - Prevent data leakage with time-based train/test splits
    - Provide feature importance rankings
    - Handle cold-start files with category-level priors
  - **Implementation Notes:** Use xgboost.XGBRegressor with isotonic calibration. Implement time-based cross-validation.
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x]* 11.1 Write unit tests for future access predictor
    - Test prediction output range
    - Test feature importance extraction
    - Test cold-start handling
    - _Requirements: 7.3, 7.5, 7.6_

  - [x]* 11.2 Write property tests for probability calibration
    - **Property 9: Model Probability Calibration (Predictor)**
    - **Validates: Requirements 7.3**

  - [x]* 11.3 Write property tests for cold-start prior application
    - **Property 17: Cold-Start Prior Application**
    - **Validates: Requirements 7.6**

- [x] 12. Tier Classification Model
  - **Phase:** 2
  - **Priority:** Critical
  - **Complexity:** Medium
  - **Dependencies:** Task 9
  - **Description:** Implement the TierClassifier XGBoost multiclass model for predicting optimal tier placement.
  - **Acceptance Criteria:**
    - Train XGBoost multiclass model (HOT, WARM, COLD)
    - Output probability distributions over tier classes
    - Provide feature importance and SHAP values
    - Support configurable hyperparameters via YAML
    - Save/load models with version identifiers
    - Generate classification reports with precision, recall, F1
  - **Implementation Notes:** Use xgboost.XGBClassifier with multi:softprob objective. Implement SHAP value computation for interpretability.
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [x]* 12.1 Write unit tests for tier classifier
    - Test probability output format
    - Test model save/load
    - Test classification report generation
    - _Requirements: 9.2, 9.5, 9.6_

  - [x]* 12.2 Write property tests for classifier probability sum
    - **Property 9: Model Probability Calibration (Classifier)**
    - **Validates: Requirements 9.2, 15.4, 16.4**

- [x] 13. Workload-ML Baseline
  - **Phase:** 2
  - **Priority:** High
  - **Complexity:** Medium
  - **Dependencies:** Task 12
  - **Description:** Implement the Workload-ML baseline classifier that uses only workload features.
  - **Acceptance Criteria:**
    - Train XGBoost classifier using only workload features
    - No semantic embeddings or future-access predictions
    - Use identical architecture as full TierClassifier
    - Output tier predictions with probability distributions
  - **Implementation Notes:** Subclass or wrap TierClassifier with feature type filtering. Ensure fair comparison by using same hyperparameters.
  - _Requirements: 15.1, 15.2, 15.3, 15.4_

  - [x]* 13.1 Write unit tests for workload-ML baseline
    - Test that only workload features are used
    - Test prediction format consistency
    - _Requirements: 15.1, 15.2, 15.4_

- [x] 14. Training Pipeline
  - **Phase:** 2
  - **Priority:** Critical
  - **Complexity:** Medium
  - **Dependencies:** Task 11, Task 12
  - **Description:** Implement the complete training pipeline for both FutureAccessPredictor and TierClassifier.
  - **Acceptance Criteria:**
    - Implement time-based train/test splits to prevent data leakage
    - Support cross-validation with temporal ordering
    - Generate tier labels from access patterns
    - Log training parameters, metrics, and model versions
    - Save trained models with version identifiers
  - **Implementation Notes:** Create `src/tiering/models/training.py`. Use sklearn's TimeSeriesSplit for cross-validation.
  - _Requirements: 7.1, 7.4, 9.1, 9.5, 21.2, 21.3_

  - [x]* 14.1 Write integration tests for training pipeline
    - Test end-to-end training workflow
    - Test temporal split enforcement
    - Test model versioning
    - _Requirements: 7.4, 9.5, 21.2_

- [x] 15. Phase 2 Checkpoint - ML Layer Validation
  - **Phase:** 2
  - **Priority:** Critical
  - **Complexity:** Small
  - **Dependencies:** Task 14
  - **Description:** Validate ML layer completion with integration tests and workload-ML baseline comparison.
  - **Acceptance Criteria:**
    - All unit and property tests pass
    - Integration test: Dataset → Features → Training → Prediction
    - Compare Workload-ML against conventional baselines
    - Document Workload-ML vs baseline results
  - **Implementation Notes:** Run experiments comparing Workload-ML to LRU, LFU, and Age baselines.
  - _Requirements: 25.2, 25.4_

- [x] 16. TF-IDF Feature Extraction
  - **Phase:** 3
  - **Priority:** High
  - **Complexity:** Medium
  - **Dependencies:** Task 2
  - **Description:** Implement the TFIDFExtractor for baseline semantic feature extraction.
  - **Acceptance Criteria:**
    - Compute TF-IDF vectors from file content and metadata
    - Apply text preprocessing: lowercasing, stopword removal, stemming
    - Produce fixed-dimension feature vectors for ML input
    - Cache computed vectors to avoid recomputation
    - Handle empty or minimal content files
  - **Implementation Notes:** Use sklearn's TfidfVectorizer. Implement caching using joblib. Store in `src/tiering/semantic/tfidf_extractor.py`.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x]* 16.1 Write unit tests for TF-IDF extraction
    - Test vector dimensions
    - Test preprocessing pipeline
    - Test empty content handling
    - _Requirements: 3.2, 3.3, 3.5_

  - [x]* 16.2 Write property tests for TF-IDF dimension consistency
    - **Property 4: Feature Vector Dimension Consistency**
    - **Validates: Requirements 3.3, 3.5**

- [x] 17. Sentence Transformer Embeddings
  - **Phase:** 3
  - **Priority:** High
  - **Complexity:** Medium
  - **Dependencies:** Task 2
  - **Description:** Implement the EmbeddingGenerator for dense document embeddings using Sentence Transformers.
  - **Acceptance Criteria:**
    - Produce dense embeddings using pre-trained Sentence Transformer models
    - Support configurable embedding dimensions via model selection
    - Store embeddings in FAISS vector store
    - Batch process files for efficient GPU utilization
    - Fall back to CPU when GPU unavailable
    - Generate consistent embeddings in deterministic mode
  - **Implementation Notes:** Use sentence-transformers library with 'all-MiniLM-L6-v2' as default model. Implement batch processing with configurable batch size.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x]* 17.1 Write unit tests for embedding generation
    - Test embedding dimensions
    - Test batch processing
    - Test CPU fallback
    - _Requirements: 4.1, 4.2, 4.4, 4.5_

  - [x]* 17.2 Write property tests for embedding dimension consistency
    - **Property 4: Feature Vector Dimension Consistency**
    - **Validates: Requirements 4.1, 4.2**

- [x] 18. FAISS Knowledge Base
  - **Phase:** 3
  - **Priority:** Critical
  - **Complexity:** Medium
  - **Dependencies:** Task 17
  - **Description:** Implement the KnowledgeBase FAISS vector store wrapper for semantic similarity search.
  - **Acceptance Criteria:**
    - Support multiple index types (Flat, IVF, HNSW)
    - Add embeddings with associated file IDs
    - Search for top-k most similar documents
    - Persist index and ID mapping to disk
    - Load existing index from disk
  - **Implementation Notes:** Use faiss-cpu (or faiss-gpu). Implement index factory pattern for index type selection.
  - _Requirements: 5.1, 5.6_

  - [x]* 18.1 Write unit tests for knowledge base
    - Test add and search operations
    - Test top-k limit enforcement
    - Test save/load persistence
    - _Requirements: 5.1, 5.6_

  - [x]* 18.2 Write property tests for retrieval limit
    - **Property 6: Knowledge Base Retrieval Limit**
    - **Validates: Requirements 5.1, 5.6**

- [x] 19. RAG Context Extraction
  - **Phase:** 3
  - **Priority:** Critical
  - **Complexity:** Large
  - **Dependencies:** Task 18
  - **Description:** Implement the RAGContextExtractor for extracting contextual features using Retrieval-Augmented Generation.
  - **Acceptance Criteria:**
    - Retrieve relevant documents from Knowledge Base using semantic search
    - Extract contextual features: category_confidence, related_files, context_summary, feature_vector
    - Output JSON validated by Pydantic RAGContext model
    - Fall back to deterministic keyword-based extraction when LLM unavailable
    - Use only local models (no OpenAI or paid API dependencies)
    - Support configurable top-k retrieval limit
    - Log all context extraction decisions
  - **Implementation Notes:** Implement LLM interface with local model support (llama.cpp, Ollama). Create deterministic fallback using TF-IDF similarity.
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 22.1, 22.2, 22.3, 22.4, 24.3_

  - [x]* 19.1 Write unit tests for RAG extraction
    - Test retrieval integration
    - Test Pydantic validation
    - Test fallback behavior
    - _Requirements: 5.1, 5.3, 5.4_

  - [x]* 19.2 Write property tests for RAG output schema
    - **Property 5: RAG Output Schema Validation**
    - **Validates: Requirements 5.3, 22.1, 22.4**

  - [x]* 19.3 Write property tests for deterministic fallback
    - **Property 18: Deterministic Fallback Behavior**
    - **Validates: Requirements 5.4, 23.1, 23.2**

- [x] 20. Semantic+Workload Classifier
  - **Phase:** 3
  - **Priority:** High
  - **Complexity:** Medium
  - **Dependencies:** Task 12, Task 16
  - **Description:** Implement the Semantic_Workload_Classifier that combines semantic and workload features without future prediction.
  - **Acceptance Criteria:**
    - Train XGBoost classifier using semantic and workload features
    - No future-access predictions
    - Use identical architecture as full TierClassifier
    - Output tier predictions with probability distributions
    - Support TF-IDF and Sentence Transformer features for ablation
  - **Implementation Notes:** Implement as feature configuration variant of TierClassifier.
  - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5_

  - [x]* 20.1 Write unit tests for semantic+workload classifier
    - Test feature type combination
    - Test ablation mode switching
    - Test prediction format
    - _Requirements: 16.1, 16.5_

- [x] 21. Phase 3 Checkpoint - Semantic Layer Validation
  - **Phase:** 3
  - **Priority:** Critical
  - **Complexity:** Small
  - **Dependencies:** Task 20
  - **Description:** Validate semantic layer completion with integration tests and classifier comparison.
  - **Acceptance Criteria:**
    - All unit and property tests pass
    - Integration test: Files → Embeddings → KB → RAG → Features → Classification
    - Compare Semantic+Workload against Workload-ML and conventional baselines
    - Document semantic contribution analysis
  - **Implementation Notes:** Run ablation experiments comparing TF-IDF vs embeddings vs RAG.
  - _Requirements: 25.5_

- [x] 22. Feature Fusion Component
  - **Phase:** 4
  - **Priority:** Critical
  - **Complexity:** Medium
  - **Dependencies:** Task 9, Task 11, Task 16, Task 19
  - **Description:** Implement the FeatureFusion component that combines workload, semantic, and future-access features.
  - **Acceptance Criteria:**
    - Concatenate workload, semantic, and future-access features into unified vectors
    - Handle missing feature types with configurable imputation
    - Apply feature scaling appropriate for XGBoost
    - Support ablation configurations to selectively disable feature types
    - Log feature contributions for analysis
  - **Implementation Notes:** Use numpy.concatenate for vector assembly. Implement standard scaler for normalization.
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x]* 22.1 Write unit tests for feature fusion
    - Test concatenation logic
    - Test missing feature imputation
    - Test ablation configuration
    - _Requirements: 8.1, 8.2, 8.4_

  - [x]* 22.2 Write property tests for fusion dimension preservation
    - **Property 13: Feature Fusion Dimension Preservation**
    - **Validates: Requirements 8.1, 8.2, 8.4**

- [x] 23. Migration Controller
  - **Phase:** 4
  - **Priority:** Critical
  - **Complexity:** Medium
  - **Dependencies:** Task 12
  - **Description:** Implement the MigrationController with policy-based migration decisions.
  - **Acceptance Criteria:**
    - Apply confidence thresholds before migrations
    - Implement hysteresis requiring sustained predictions
    - Enforce cooldown periods between migrations
    - Model migration costs (latency, bandwidth)
    - Log all migration decisions with rationale
    - Support configurable policies via YAML
  - **Implementation Notes:** Track per-file state for hysteresis and cooldown. Implement cost calculation model.
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [x]* 23.1 Write unit tests for migration controller
    - Test confidence threshold application
    - Test hysteresis logic
    - Test cooldown enforcement
    - _Requirements: 10.1, 10.2, 10.3_

  - [x]* 23.2 Write property tests for migration confidence threshold
    - **Property 10: Migration Confidence Threshold**
    - **Validates: Requirements 10.1**

  - [x]* 23.3 Write property tests for migration hysteresis
    - **Property 11: Migration Hysteresis Enforcement**
    - **Validates: Requirements 10.2**

  - [x]* 23.4 Write property tests for migration cooldown
    - **Property 12: Migration Cooldown Enforcement**
    - **Validates: Requirements 10.3**

- [x] 24. Full Context-Aware Classifier
  - **Phase:** 4
  - **Priority:** Critical
  - **Complexity:** Medium
  - **Dependencies:** Task 22, Task 23
  - **Description:** Implement the complete Context_Aware_Classifier integrating all feature types and migration control.
  - **Acceptance Criteria:**
    - Train XGBoost classifier using workload, semantic, and future-access features
    - Support optional RAG context features
    - Integrate with MigrationController for policy-based transitions
    - Use identical model architecture as baselines
    - Prevent data leakage in all feature computation
  - **Implementation Notes:** This is the full system combining all components. Ensure architecture parity with baselines.
  - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5_

  - [x]* 24.1 Write unit tests for context-aware classifier
    - Test feature integration
    - Test migration controller integration
    - Test RAG feature toggling
    - _Requirements: 17.1, 17.2, 17.3_

- [x] 25. Integration Tests for Full Pipeline
  - **Phase:** 4
  - **Priority:** Critical
  - **Complexity:** Medium
  - **Dependencies:** Task 24
  - **Description:** Implement comprehensive integration tests for the complete system pipeline.
  - **Acceptance Criteria:**
    - Test: Dataset → Workload → Features → Training → Classification
    - Test: Full simulation with context-aware classifier
    - Test: Baseline comparison on same data
    - Test: Data leakage prevention across pipeline
    - Validate all components work together correctly
  - **Implementation Notes:** Use pytest fixtures for shared test data. Create integration test scenarios covering main workflows.
  - _Requirements: 21.1, 21.2, 21.3, 21.4, 21.5_

  - [x]* 25.1 Write integration tests for training pipeline
    - Test end-to-end training with all feature types
    - _Requirements: 7.1, 9.1, 21.2_

  - [x]* 25.2 Write integration tests for simulation pipeline
    - Test full simulation run with context-aware classifier
    - _Requirements: 11.1, 11.2, 11.4_

- [x] 26. Phase 4 Checkpoint - Full System Validation
  - **Phase:** 4
  - **Priority:** Critical
  - **Complexity:** Small
  - **Dependencies:** Task 25
  - **Description:** Validate full system integration with comprehensive comparison against all baselines.
  - **Acceptance Criteria:**
    - All tests pass (unit, property, integration)
    - Run full comparison: Context-Aware vs all baselines
    - Verify metrics improvement over baselines
    - Document full system validation results
  - **Implementation Notes:** This is the key validation checkpoint before experiments phase.
  - _Requirements: 25.5_

- [x] 27. Metrics Collection System
  - **Phase:** 5
  - **Priority:** Critical
  - **Complexity:** Medium
  - **Dependencies:** Task 7
  - **Description:** Implement comprehensive MetricsCollector for classification, cost, latency, and migration metrics.
  - **Acceptance Criteria:**
    - Compute classification metrics: accuracy, precision, recall, F1 per class, confusion matrices
    - Compute storage cost metrics: total cost, cost per tier, cost savings vs baseline
    - Compute latency metrics: average, 95th percentile, per tier
    - Compute migration metrics: total migrations, overhead, thrashing rate
    - Output metrics in JSON format with timestamps
    - Generate visualization plots for key metrics
  - **Implementation Notes:** Use sklearn.metrics for classification metrics. Use matplotlib/seaborn for visualizations.
  - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6_

  - [x]* 27.1 Write unit tests for metrics collection
    - Test metric calculation correctness
    - Test JSON output format
    - Test visualization generation
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5_

  - [x]* 27.2 Write property tests for metrics computation
    - **Property 20: Metrics Computation Correctness**
    - **Validates: Requirements 11.4, 19.1, 19.2, 19.3, 19.4**

- [x] 28. Experiment Framework
  - **Phase:** 5
  - **Priority:** Critical
  - **Complexity:** Large
  - **Dependencies:** Task 26, Task 27
  - **Description:** Implement the automated Experiment_Framework for systematic tiering approach comparison.
  - **Acceptance Criteria:**
    - Execute experiments defined in YAML configuration files
    - Support six experiment types: baseline comparison, workload ML, semantic ablation, RAG ablation, future prediction ablation, migration policy stress test
    - Record all parameters, seeds, and model versions
    - Output results in structured JSON format
    - Support parallel execution for independent experiments
    - Generate comparison reports and visualizations
  - **Implementation Notes:** Implement as orchestrator class that configures and runs experiments. Use multiprocessing for parallel execution.
  - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6_

  - [x]* 28.1 Write unit tests for experiment framework
    - Test experiment configuration loading
    - Test result output format
    - Test parallel execution
    - _Requirements: 18.1, 18.3, 18.4, 18.5_

- [x] 29. Automated Experiment Runners
  - **Phase:** 5
  - **Priority:** High
  - **Complexity:** Medium
  - **Dependencies:** Task 28
  - **Description:** Implement specific experiment runner classes for each experiment type.
  - **Acceptance Criteria:**
    - Implement ConventionalBaselineExperiment runner
    - Implement WorkloadMLComparisonExperiment runner
    - Implement SemanticAblationExperiment runner
    - Implement RAGAblationExperiment runner
    - Implement FuturePredictionAblationExperiment runner
    - Implement MigrationPolicyStressTest runner
  - **Implementation Notes:** Each runner should be a callable that takes configuration and returns experiment results.
  - _Requirements: 18.2_

  - [x]* 29.1 Write unit tests for experiment runners
    - Test each runner with minimal configuration
    - Test result format consistency
    - _Requirements: 18.2_

- [x] 30. Visualization and Reporting
  - **Phase:** 5
  - **Priority:** High
  - **Complexity:** Medium
  - **Dependencies:** Task 27, Task 28
  - **Description:** Implement visualization and reporting components for experiment results.
  - **Acceptance Criteria:**
    - Generate comparison plots for all approaches
    - Generate confusion matrix visualizations
    - Generate feature importance plots (XGBoost and SHAP)
    - Generate cost and latency distribution plots
    - Generate ablation study comparison charts
    - Output reports in markdown and HTML formats
  - **Implementation Notes:** Use matplotlib and seaborn for plotting. Create report templates using Jinja2 or similar.
  - _Requirements: 18.6, 19.6_

  - [x]* 30.1 Write unit tests for visualization
    - Test plot generation
    - Test report format
    - _Requirements: 18.6, 19.6_

- [x] 31. Ablation Study Framework
  - **Phase:** 5
  - **Priority:** High
  - **Complexity:** Medium
  - **Dependencies:** Task 28
  - **Description:** Implement framework for running and analyzing ablation studies.
  - **Acceptance Criteria:**
    - Support disabling individual feature types (workload, semantic, future)
    - Support disabling RAG context
    - Compare ablated results against full system
    - Quantify contribution of each feature type
    - Generate ablation analysis reports
  - **Implementation Notes:** Use FeatureFusion ablation configuration. Run multiple experiments with different feature combinations.
  - _Requirements: 8.4, 16.5, 17.2_

  - [x]* 31.1 Write unit tests for ablation framework
    - Test feature type toggling
    - Test contribution analysis
    - _Requirements: 8.4_

- [x] 32. Data Leakage Validation System
  - **Phase:** 5
  - **Priority:** Critical
  - **Complexity:** Medium
  - **Dependencies:** Task 25
  - **Description:** Implement the Validation_System for detecting and flagging potential data leakage.
  - **Acceptance Criteria:**
    - Detect temporal violations in feature computation
    - Flag training data that includes future information
    - Validate proper time-based splits in model training
    - Halt execution with descriptive errors when leakage detected
  - **Implementation Notes:** Implement as validation pass that checks feature timestamps against reference times.
  - _Requirements: 21.5, 21.6_

  - [x]* 32.1 Write unit tests for leakage detection
    - Test detection of temporal violations
    - Test proper error messages
    - _Requirements: 21.5, 21.6_

- [x] 33. Final Checkpoint - Full System Validation
  - **Phase:** 5
  - **Priority:** Critical
  - **Complexity:** Small
  - **Dependencies:** Task 31, Task 32
  - **Description:** Final validation of the complete system with all experiments running successfully.
  - **Acceptance Criteria:**
    - All tests pass across all phases
    - All six experiment types run successfully
    - Generate final comparison report
    - Document all results and findings
    - Validate reproducibility with different seeds
  - **Implementation Notes:** This is the final validation before the system is ready for research use.
  - _Requirements: 25.5_

- [x] 34. README Documentation
  - **Phase:** All
  - **Priority:** High
  - **Complexity:** Small
  - **Dependencies:** Task 33
  - **Description:** Create comprehensive README documentation for the repository.
  - **Acceptance Criteria:**
    - Setup instructions with dependency installation
    - Quick start guide for running experiments
    - Architecture overview with component diagram
    - Configuration guide with example YAML files
    - Experiment execution guide
    - Result interpretation guide
  - **Implementation Notes:** Include badges for test status, coverage, and python version support.
  - _Requirements: 26.7_

- [x] 35. API Documentation
  - **Phase:** All
  - **Priority:** Medium
  - **Complexity:** Medium
  - **Dependencies:** Task 34
  - **Description:** Generate comprehensive API documentation for all public interfaces.
  - **Acceptance Criteria:**
    - Document all public classes and methods
    - Include type hints and docstrings
    - Generate HTML documentation using Sphinx or pdoc
    - Include usage examples for key components
  - **Implementation Notes:** Use docstrings in Google or NumPy style. Generate docs with Sphinx.
  - _Requirements: 26.7_

---

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at each phase
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- Documentation tasks are for the complete system after Phase 5

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "2.2", "2.3"] },
    { "id": 1, "tasks": ["3.1", "3.2", "4.1", "5.1", "6.1"] },
    { "id": 2, "tasks": ["7.1", "7.2", "7.3", "9.1", "9.2", "9.3"] },
    { "id": 3, "tasks": ["10.1", "10.2", "11.1", "11.2", "11.3", "12.1", "12.2"] },
    { "id": 4, "tasks": ["13.1", "14.1", "16.1", "16.2", "17.1", "17.2"] },
    { "id": 5, "tasks": ["18.1", "18.2", "19.1", "19.2", "19.3", "20.1"] },
    { "id": 6, "tasks": ["22.1", "22.2", "23.1", "23.2", "23.3", "23.4", "24.1"] },
    { "id": 7, "tasks": ["25.1", "25.2", "27.1", "27.2", "28.1"] },
    { "id": 8, "tasks": ["29.1", "30.1", "31.1", "32.1"] }
  ]
}
```
