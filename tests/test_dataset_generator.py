"""Unit tests for DatasetGenerator."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.tiering.ingestion.dataset_generator import DatasetGenerator
from src.tiering.ingestion.schemas import Category, FileDocument


class TestDatasetGenerator:
    """Test suite for DatasetGenerator."""
    
    def test_generate_creates_files(self):
        """Test that generate() creates the correct number of files."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=100)
        
        assert len(files) == 100
        assert all(isinstance(f, FileDocument) for f in files)
    
    def test_generate_creates_all_categories(self):
        """Test that generate() creates files across all nine categories."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=450)  # 50 per category
        
        categories = {f.category for f in files}
        expected_categories = {c.value for c in Category}
        
        assert categories == expected_categories
    
    def test_file_id_uniqueness(self):
        """Test that all generated file IDs are unique."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=1000)
        
        file_ids = [f.file_id for f in files]
        assert len(file_ids) == len(set(file_ids)), "File IDs must be unique"
    
    def test_file_id_format(self):
        """Test that file IDs are valid UUIDs."""
        from uuid import UUID
        
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=10)
        
        for file_doc in files:
            # Should not raise ValueError if valid UUID
            UUID(file_doc.file_id)
    
    def test_filenames_generated(self):
        """Test that filenames are generated correctly."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=100)
        
        for file_doc in files:
            assert file_doc.filename, f"Filename missing for file {file_doc.file_id}"
            assert any(
                file_doc.filename.endswith(ext)
                for ext in [".pdf", ".docx", ".xlsx", ".txt", ".json", ".csv"]
            ), f"Filename {file_doc.filename} missing expected extension"
    
    def test_content_generated(self):
        """Test that content is generated for all files."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=100)
        
        for file_doc in files:
            assert len(file_doc.content) > 0, f"Empty content for file {file_doc.file_id}"
            assert isinstance(file_doc.content, str)
    
    def test_size_within_range(self):
        """Test that file sizes are within expected ranges."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=100)
        
        for file_doc in files:
            category_templates = generator._CATEGORY_TEMPLATES[Category(file_doc.category)]
            min_size, max_size = category_templates["size_range"]
            
            assert min_size <= file_doc.size_bytes <= max_size, (
                f"Size {file_doc.size_bytes} out of range [{min_size}, {max_size}] "
                f"for category {file_doc.category}"
            )
    
    def test_timestamps_valid(self):
        """Test that timestamps are valid and ordered correctly."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=100)
        
        for file_doc in files:
            assert isinstance(file_doc.created_at, datetime)
            assert isinstance(file_doc.modified_at, datetime)
            # Modified time should be >= created time
            assert file_doc.modified_at >= file_doc.created_at, (
                f"Modified time {file_doc.modified_at} before created time {file_doc.created_at}"
            )
            # Both should be in 2024
            assert file_doc.created_at.year == 2024
            assert file_doc.modified_at.year == 2024
    
    def test_metadata_fields(self):
        """Test that metadata contains expected fields."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=10)
        
        expected_fields = {"category", "file_type", "department", "confidentiality", 
                          "indexed", "compressed", "checksum"}
        
        for file_doc in files:
            assert set(file_doc.metadata.keys()) == expected_fields, (
                f"Metadata fields mismatch: expected {expected_fields}, "
                f"got {set(file_doc.metadata.keys())}"
            )
    
    def test_reproducibility_same_seed(self):
        """Test that identical seeds produce identical output."""
        gen1 = DatasetGenerator(seed=42)
        files1 = gen1.generate(num_files=50)
        
        gen2 = DatasetGenerator(seed=42)
        files2 = gen2.generate(num_files=50)
        
        assert len(files1) == len(files2)
        
        for f1, f2 in zip(files1, files2):
            assert f1.file_id == f2.file_id
            assert f1.filename == f2.filename
            assert f1.category == f2.category
            assert f1.content == f2.content
            assert f1.size_bytes == f2.size_bytes
            assert f1.created_at == f2.created_at
            assert f1.modified_at == f2.modified_at
            assert f1.metadata == f2.metadata
    
    def test_reproducibility_different_seed(self):
        """Test that different seeds produce different output."""
        gen1 = DatasetGenerator(seed=42)
        files1 = gen1.generate(num_files=50)
        
        gen2 = DatasetGenerator(seed=43)
        files2 = gen2.generate(num_files=50)
        
        # Files should be different
        different_count = sum(
            1 for f1, f2 in zip(files1, files2)
            if f1.file_id != f2.file_id or f1.filename != f2.filename
        )
        
        assert different_count > 0, "Different seeds should produce different results"
    
    def test_category_distribution_default(self):
        """Test default uniform category distribution."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=900)  # 100 per category
        
        category_counts = {}
        for file_doc in files:
            category = file_doc.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Should have roughly equal distribution (with rounding variance)
        for category in Category:
            count = category_counts.get(category.value, 0)
            # Allow ±20% variance due to rounding
            assert 80 <= count <= 120, (
                f"Category {category.value} has {count} files, expected ~100"
            )
    
    def test_category_distribution_custom(self):
        """Test custom category distribution."""
        config = {
            "category_distribution": {
                "Finance": 50,  # Heavy weight
                "Legal": 1,
                "HR": 1,
                "Engineering": 1,
                "Research": 1,
                "Marketing": 1,
                "Operations": 1,
                "Security": 1,
                "System": 1,
            }
        }
        
        generator = DatasetGenerator(config=config, seed=42)
        files = generator.generate(num_files=60)
        
        finance_count = sum(1 for f in files if f.category == "Finance")
        others_count = len(files) - finance_count
        
        # Finance should be majority
        assert finance_count > others_count
    
    def test_jsonl_output_format(self):
        """Test that to_jsonl() produces valid JSON Lines format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dataset.jsonl"
            
            generator = DatasetGenerator(seed=42)
            generator.generate(num_files=10)
            generator.to_jsonl(output_path)
            
            assert output_path.exists()
            
            # Read and validate JSONL format
            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 10
            
            for line in lines:
                data = json.loads(line)
                # Should be able to reconstruct FileDocument
                doc = FileDocument(**data)
                assert doc.file_id
                assert doc.filename
                assert doc.category
    
    def test_jsonl_schema_validation(self):
        """Test that JSON Lines output passes schema validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dataset.jsonl"
            
            generator = DatasetGenerator(seed=42)
            generator.generate(num_files=20)
            generator.to_jsonl(output_path)
            
            # Read and validate each line
            with open(output_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    # This will raise ValidationError if invalid
                    FileDocument(**data)
    
    def test_jsonl_creates_parent_directories(self):
        """Test that to_jsonl() creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "dataset.jsonl"
            
            generator = DatasetGenerator(seed=42)
            generator.generate(num_files=5)
            generator.to_jsonl(output_path)
            
            assert output_path.exists()
            assert output_path.parent.exists()
    
    def test_zero_files(self):
        """Test generating zero files."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=0)
        
        assert len(files) == 0
    
    def test_small_dataset(self):
        """Test generating small dataset with fewer files than categories."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=5)
        
        assert len(files) == 5
        # Should still have valid files
        for file_doc in files:
            assert file_doc.file_id
            assert file_doc.filename
    
    def test_large_dataset(self):
        """Test generating larger dataset."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=10000)
        
        assert len(files) == 10000
        assert len({f.file_id for f in files}) == 10000
        
        # All categories should be represented
        categories = {f.category for f in files}
        assert len(categories) == 9
    
    def test_category_appropriate_sizes(self):
        """Test that different categories have appropriate size distributions."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=900)  # 100 per category
        
        # Group by category
        by_category = {}
        for f in files:
            if f.category not in by_category:
                by_category[f.category] = []
            by_category[f.category].append(f.size_bytes)
        
        # Engineering should have larger files than HR on average
        eng_avg = sum(by_category.get("Engineering", [0])) / max(len(by_category.get("Engineering", [1])), 1)
        hr_avg = sum(by_category.get("HR", [0])) / max(len(by_category.get("HR", [1])), 1)
        
        assert eng_avg > hr_avg
    
    def test_file_document_pydantic_validation(self):
        """Test that FileDocument Pydantic model validates properly."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=1)
        
        file_doc = files[0]
        
        # Should be able to serialize and deserialize
        as_dict = file_doc.model_dump()
        as_json = file_doc.model_dump_json()
        assert as_dict["file_id"] == file_doc.file_id
        
        reconstructed = FileDocument(**json.loads(as_json))
        assert reconstructed.file_id == file_doc.file_id


class TestDatasetGeneratorProperties:
    """Property-based tests for DatasetGenerator (2.2 Write property tests for dataset reproducibility)."""
    
    def test_reproducibility_with_various_seeds(self):
        """Property: Different runs with same seed produce identical files."""
        test_seeds = [0, 42, 100, 999, 12345]
        
        for seed in test_seeds:
            gen1 = DatasetGenerator(seed=seed)
            files1 = gen1.generate(num_files=100)
            
            gen2 = DatasetGenerator(seed=seed)
            files2 = gen2.generate(num_files=100)
            
            # Exact equality check
            assert len(files1) == len(files2)
            for f1, f2 in zip(files1, files2):
                assert f1.model_dump() == f2.model_dump(), (
                    f"Files differ for seed {seed}: {f1.file_id} vs {f2.file_id}"
                )
    
    def test_determinism_with_fixed_count(self):
        """Property: Same generator with same seed always produces same output."""
        for num_files in [1, 10, 100, 1000]:
            gen = DatasetGenerator(seed=42)
            result1 = gen.generate(num_files=num_files)
            
            gen = DatasetGenerator(seed=42)
            result2 = gen.generate(num_files=num_files)
            
            assert len(result1) == len(result2) == num_files
            assert all(f1.model_dump() == f2.model_dump() for f1, f2 in zip(result1, result2))


class TestDatasetGeneratorSchemaCompliance:
    """Property-based tests for dataset schema compliance (2.3)."""
    
    def test_schema_compliance_all_categories(self):
        """Property: All generated files have valid schema for their category."""
        generator = DatasetGenerator(seed=42)
        files = generator.generate(num_files=450)  # 50 per category
        
        for file_doc in files:
            # Validate all required fields exist
            assert file_doc.file_id
            assert file_doc.filename
            assert file_doc.category in {c.value for c in Category}
            assert file_doc.content
            assert file_doc.metadata
            assert file_doc.size_bytes >= 0
            assert isinstance(file_doc.created_at, datetime)
            assert isinstance(file_doc.modified_at, datetime)
            
            # Validate metadata structure
            assert "category" in file_doc.metadata
            assert "file_type" in file_doc.metadata
            assert "department" in file_doc.metadata
            assert "confidentiality" in file_doc.metadata
            assert "indexed" in file_doc.metadata
            assert "compressed" in file_doc.metadata
            assert "checksum" in file_doc.metadata
    
    def test_jsonl_schema_compliance(self):
        """Property: All JSON Lines output conforms to FileDocument schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dataset.jsonl"
            
            generator = DatasetGenerator(seed=42)
            generator.generate(num_files=100)
            generator.to_jsonl(output_path)
            
            with open(output_path, "r") as f:
                for line in f:
                    # Should parse and validate without error
                    data = json.loads(line)
                    doc = FileDocument(**data)
                    
                    # Verify schema compliance
                    assert doc.file_id
                    assert 0 <= doc.size_bytes
                    assert doc.created_at.year == 2024
                    assert doc.modified_at.year == 2024
