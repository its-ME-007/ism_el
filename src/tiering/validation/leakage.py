"""Temporal-causality checks for features and training splits."""


class DataLeakageError(ValueError):
    """Raised when future information enters a historical prediction."""


class DataLeakageValidator:
    @staticmethod
    def validate_feature_timestamps(feature_timestamps, reference_time):
        violations = [timestamp for timestamp in feature_timestamps
                      if timestamp is not None and timestamp > reference_time]
        if violations:
            earliest = min(violations)
            raise DataLeakageError(
                f"Feature timestamp {earliest.isoformat()} is later than decision time "
                f"{reference_time.isoformat()}"
            )
        return True

    @staticmethod
    def validate_temporal_split(train_timestamps, test_timestamps):
        if not train_timestamps or not test_timestamps:
            raise ValueError("Training and test timestamps must both be non-empty")
        latest_train = max(train_timestamps)
        earliest_test = min(test_timestamps)
        if latest_train >= earliest_test:
            raise DataLeakageError(
                f"Temporal split overlaps: latest training time {latest_train.isoformat()} "
                f"is not earlier than earliest test time {earliest_test.isoformat()}"
            )
        return True

    @classmethod
    def validate_training_samples(cls, samples, reference_times):
        if len(samples) != len(reference_times):
            raise ValueError("Each training sample must have a reference time")
        for sample, reference_time in zip(samples, reference_times):
            cls.validate_feature_timestamps(sample, reference_time)
        return True
