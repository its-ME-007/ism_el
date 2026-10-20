"""Synthetic dataset generation for enterprise file storage."""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from .schemas import Category, FileDocument


class DatasetGenerator:
    """
    Generates synthetic enterprise datasets with realistic category distributions.
    
    Creates files across nine enterprise categories with category-specific:
    - File names and naming patterns
    - Content snippets
    - Realistic file sizes
    - Metadata
    
    All generation is reproducible with deterministic seeds.
    """
    
    # Category-specific templates and generation rules
    _CATEGORY_TEMPLATES = {
        Category.FINANCE: {
            "name_patterns": [
                "Q{q}_financial_report_{suffix}",
                "budget_allocation_{suffix}",
                "expense_report_{date}",
                "revenue_forecast_{quarter}_{year}",
                "annual_audit_{year}",
                "tax_return_{year}",
                "invoice_{invoice_id}",
                "payment_receipt_{date}",
            ],
            "content_samples": [
                "Financial Statement for Q{} 2024: Revenue ${:.2f}M, Expenses ${:.2f}M",
                "Budget Allocation: Salaries 45%, Operations 30%, Marketing 15%, R&D 10%",
                "Expense Report {} - Categories: Travel ${}, Meals ${}, Equipment ${}",
                "Revenue Forecast {} - Conservative: ${:.1f}M, Optimistic: ${:.1f}M",
                "Annual Audit Report: Total Assets ${:.2f}M, Liabilities ${:.2f}M",
                "Tax Return Filing: Income ${:.2f}M, Deductions ${:.2f}M, Tax ${:.2f}M",
                "Invoice {}: Amount ${}, Terms Net-30, Due Date {}",
                "Payment Receipt {}: Transaction ID {}, Amount ${}, Status Complete",
            ],
            "size_range": (50_000, 5_000_000),  # 50KB to 5MB
        },
        Category.LEGAL: {
            "name_patterns": [
                "contract_agreement_{client}",
                "nda_{date}",
                "terms_conditions_v{version}",
                "litigation_case_{case_id}",
                "compliance_checklist_{year}",
                "privacy_policy_{version}",
                "legal_opinion_{date}",
                "settlement_agreement_{case_id}",
            ],
            "content_samples": [
                "CONTRACT AGREEMENT between {} and {} dated {}",
                "Non-Disclosure Agreement: Confidential Information restricted until {}",
                "Terms & Conditions v{}: Liability limitation to ${}, Governing law: {}",
                "Litigation Case {}: Filed {}, Judge: {}, Status: {}",
                "Compliance Checklist {}: GDPR [✓], CCPA [✓], HIPAA [{}]",
                "Privacy Policy v{}: Data Retention {} days, User Rights: {}",
                "Legal Opinion on {}: Conclusion favorable, Risk level low to moderate",
                "Settlement Agreement Case {}: Amount ${}, Payment terms: {}",
            ],
            "size_range": (100_000, 10_000_000),  # 100KB to 10MB
        },
        Category.HR: {
            "name_patterns": [
                "employee_roster_{date}",
                "payroll_{month}_{year}",
                "training_records_{employee_id}",
                "performance_review_{quarter}_{year}",
                "recruitment_pipeline_{date}",
                "benefits_enrollment_{year}",
                "employee_handbook_v{version}",
                "attendance_log_{month}",
            ],
            "content_samples": [
                "Employee Roster {}: Total Headcount {}, New Hires {}, Departures {}",
                "Payroll Summary {} {}: Gross Payroll ${:.2f}M, Tax Withholding ${:.2f}M",
                "Training Records Employee {}: Completed courses: {}, Hours: {}",
                "Performance Review Q{} {}: Rating {} / 5.0, Feedback: {}",
                "Recruitment Pipeline {}: Active postings {}, Applications {}, Interviews {}",
                "Benefits Enrollment {}: Health insurance {}, 401k participation {}%",
                "Employee Handbook v{}: {} pages, Last updated {}, Coverage: {}",
                "Attendance Log {}: Present {}, Absent {}, Late {}, Sick {}",
            ],
            "size_range": (30_000, 2_000_000),  # 30KB to 2MB
        },
        Category.ENGINEERING: {
            "name_patterns": [
                "source_code_repository_{branch}",
                "build_artifacts_v{version}",
                "test_results_{date}",
                "deployment_logs_{environment}",
                "architecture_documentation_{component}",
                "code_review_{pr_id}",
                "database_schema_{version}",
                "api_specification_v{version}",
            ],
            "content_samples": [
                "Source Code Repository: {} files, {} lines of code, {} commits",
                "Build Artifacts v{}: {} jar files, {} dependencies, Build time {} seconds",
                "Test Results {}: {} tests, {} passed, {} failed, Coverage {}%",
                "Deployment Log {}: Started {}, Completed {}, Status: {}",
                "Architecture: {} microservices, {} databases, {} message queues",
                "Code Review PR{}: {} files changed, {} additions, {} deletions",
                "Database Schema v{}: {} tables, {} procedures, Last modified {}",
                "API Specification v{}: {} endpoints, Latency p99: {}ms, Uptime: {}%",
            ],
            "size_range": (1_000_000, 100_000_000),  # 1MB to 100MB
        },
        Category.RESEARCH: {
            "name_patterns": [
                "research_paper_{year}_{topic}",
                "dataset_{research_id}",
                "experiment_results_{trial_id}",
                "literature_review_{topic}",
                "research_proposal_{grant_id}",
                "methodology_{study_id}",
                "raw_data_{experiment_id}",
                "analysis_report_{project_id}",
            ],
            "content_samples": [
                "Research Paper {}: Authors {}, Published {}, Citations {}",
                "Dataset {}: {} records, {} features, Size {} MB",
                "Experiment Results Trial {}: Hypothesis confirmed, p-value {}, CI [{}]",
                "Literature Review {}: {} papers reviewed, {} key findings",
                "Research Proposal Grant {}: Budget ${}, Duration {} months, Team size {}",
                "Methodology: {} subjects, {} sessions, Protocol reviewed {}",
                "Raw Data {}: {} observations, {} variables, Quality score {}%",
                "Analysis Report {}: {} figures, {} tables, {} conclusions",
            ],
            "size_range": (500_000, 50_000_000),  # 500KB to 50MB
        },
        Category.MARKETING: {
            "name_patterns": [
                "campaign_report_{campaign_id}",
                "customer_analytics_{date}",
                "marketing_plan_{quarter}_{year}",
                "social_media_content_{platform}",
                "email_template_{template_id}",
                "brand_guidelines_v{version}",
                "market_research_{topic}",
                "advertising_spend_{month}",
            ],
            "content_samples": [
                "Campaign Report {}: Impressions {}, Clicks {}, CTR {}%, Conversions {}",
                "Customer Analytics {}: Total customers {}, Active {}, Churn rate {}%",
                "Marketing Plan Q{} {}: Budget ${:.2f}M, Channels: {}, Target: {}",
                "Social Content {}: {} posts, {} engagement rate, {} new followers",
                "Email Template {}: Subject line: {}, Open rate {}%, Click rate {}%",
                "Brand Guidelines v{}: Logo usage, Color palette, Typography",
                "Market Research {}: {} respondents, {} key insights, {} opportunities",
                "Advertising Spend {}: Google Ads ${}, Facebook ${}, LinkedIn ${}",
            ],
            "size_range": (100_000, 3_000_000),  # 100KB to 3MB
        },
        Category.OPERATIONS: {
            "name_patterns": [
                "inventory_report_{location}",
                "supply_chain_status_{date}",
                "facility_maintenance_{facility_id}",
                "vendor_agreements_{vendor_id}",
                "shipping_logs_{date}",
                "operational_metrics_{month}",
                "process_documentation_{process_id}",
                "incident_report_{incident_id}",
            ],
            "content_samples": [
                "Inventory Report {}: Total items {}, Stock level {}%, Reorder items {}",
                "Supply Chain Status {}: {} suppliers active, Delivery time {} days",
                "Facility Maintenance {}: {} items serviced, Uptime {}%, Cost ${}",
                "Vendor Agreement {}: Payment terms {}, Discount {}, Contract expires {}",
                "Shipping Logs {}: {} packages, On-time delivery {}%, Carrier: {}",
                "Operational Metrics {}: Efficiency {}%, Downtime {} hours, Cost ${:.2f}M",
                "Process Documentation {}: {} steps, Cycle time {} hours, Owner: {}",
                "Incident Report {}: Severity {}, Duration {} minutes, Root cause: {}",
            ],
            "size_range": (50_000, 2_000_000),  # 50KB to 2MB
        },
        Category.SECURITY: {
            "name_patterns": [
                "access_log_{date}",
                "vulnerability_scan_{scan_id}",
                "incident_response_{incident_id}",
                "firewall_rules_v{version}",
                "security_audit_{audit_id}",
                "threat_intelligence_{date}",
                "encryption_keys_{environment}",
                "compliance_report_{standard}",
            ],
            "content_samples": [
                "Access Log {}: {} login attempts, {} failed, {} blocked by security",
                "Vulnerability Scan {}: {} vulnerabilities, {} critical, {} medium, {} low",
                "Incident Response {}: Detected {}, Contained {}, Resolution time {} hours",
                "Firewall Rules v{}: {} rules, {} IPs blocked, {} allowed",
                "Security Audit {}: {} findings, {} critical, Remediation timeline {}",
                "Threat Intelligence {}: {} threats detected, {} indicators of compromise",
                "Encryption Keys {}: {} keys active, {} rotated this month",
                "Compliance Report {}: Passes {}, Fails {}, Score {}%",
            ],
            "size_range": (100_000, 5_000_000),  # 100KB to 5MB
        },
        Category.SYSTEM: {
            "name_patterns": [
                "system_log_{date}",
                "backup_archive_{backup_id}",
                "performance_metrics_{host}",
                "configuration_baseline_{version}",
                "disk_usage_report_{date}",
                "network_traffic_analysis_{date}",
                "patch_history_{month}",
                "system_health_check_{date}",
            ],
            "content_samples": [
                "System Log {}: {} events, {} errors, {} warnings, Average latency {} ms",
                "Backup Archive {}: Size {} GB, Compressed {} GB, Stored {}",
                "Performance Metrics {}: CPU {}%, Memory {}%, Disk {}%, Network {} Mbps",
                "Configuration Baseline v{}: {} parameters, Last updated {}",
                "Disk Usage Report {}: Total {} TB, Used {} TB, Free {} TB",
                "Network Traffic {}: {} packets, {} MB transferred, {} connections",
                "Patch History {}: {} patches applied, {} critical, {} security updates",
                "System Health: Uptime {}%, Services {} running, Alerts {} active",
            ],
            "size_range": (200_000, 20_000_000),  # 200KB to 20MB
        },
    }
    
    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        """
        Initialize generator with configuration and random seed.
        
        Args:
            config: Configuration dict with optional keys:
                - category_distribution: Dict[Category, float] - relative weights
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.uuid_counter = 0  # Counter for deterministic UUID generation
        
        # Set default category distribution (uniform)
        self.category_distribution = {
            category: 1.0 for category in Category
        }
        
        # Override with config if provided
        if config and "category_distribution" in config:
            dist = config["category_distribution"]
            for category in Category:
                if category.value in dist:
                    self.category_distribution[category] = dist[category.value]
        
        self.files: List[FileDocument] = []
    
    def _generate_deterministic_uuid(self) -> str:
        """
        Generate a deterministic UUID based on seed and counter.
        
        This ensures reproducibility: same seed always generates same UUIDs
        in the same order.
        """
        # Use counter-based UUID generation for determinism
        # Create a UUID from hash of seed + counter
        import hashlib
        hash_input = f"{self.seed}_{self.uuid_counter}".encode()
        hash_value = hashlib.md5(hash_input).hexdigest()
        
        # Create a UUID from the hash (set version 3 for MD5-based)
        uuid_str = f"{hash_value[:8]}-{hash_value[8:12]}-3{hash_value[13:16]}-{hash_value[16:20]}-{hash_value[20:32]}"
        self.uuid_counter += 1
        
        return uuid_str
    
    def generate(self, num_files: int) -> List[FileDocument]:
        """
        Generate synthetic files across enterprise categories.
        
        Args:
            num_files: Number of files to generate
            
        Returns:
            List of FileDocument instances with unique IDs, categories, content, metadata
        """
        self.files = []
        
        # Calculate number of files per category based on distribution
        total_weight = sum(self.category_distribution.values())
        normalized_dist = {
            cat: weight / total_weight
            for cat, weight in self.category_distribution.items()
        }
        
        files_per_category = {}
        remaining = num_files
        
        for category in Category:
            count = int(normalized_dist[category] * num_files)
            files_per_category[category] = count
            remaining -= count
        
        # Distribute remaining files (due to rounding) to first categories
        for category in Category:
            if remaining > 0:
                files_per_category[category] += 1
                remaining -= 1
        
        # Generate files for each category
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        
        for category, count in files_per_category.items():
            for i in range(count):
                file_id = self._generate_deterministic_uuid()
                filename = self._generate_filename(category, i)
                content = self._generate_content(category, i)
                size = self._generate_size(category)
                
                # Vary creation times across the year
                days_offset = self.rng.randint(0, 365)
                created_at = base_time + timedelta(days=days_offset)
                
                # Modified time is after or equal to created time
                modified_days_offset = self.rng.randint(days_offset, 365)
                modified_at = base_time + timedelta(days=modified_days_offset)
                
                metadata = self._generate_metadata(category, i)
                
                file_doc = FileDocument(
                    file_id=file_id,
                    filename=filename,
                    category=category,
                    content=content,
                    metadata=metadata,
                    size_bytes=size,
                    created_at=created_at,
                    modified_at=modified_at,
                )
                
                self.files.append(file_doc)
        
        return self.files
    
    def _generate_filename(self, category: Category, index: int) -> str:
        """Generate realistic filename for category."""
        templates = self._CATEGORY_TEMPLATES[category]["name_patterns"]
        template = self.rng.choice(templates)
        
        # Fill template with random values
        replacements = {
            "{q}": str(self.rng.randint(1, 4)),
            "{suffix}": f"{self.rng.randint(1000, 9999)}",
            "{date}": f"{self.rng.randint(1, 28)}-{self.rng.randint(1, 12)}-2024",
            "{quarter}": f"Q{self.rng.randint(1, 4)}",
            "{year}": "2024",
            "{invoice_id}": f"INV-{self.rng.randint(10000, 99999)}",
            "{version}": str(self.rng.randint(1, 10)),
            "{client}": self.rng.choice(["Acme Corp", "TechCo", "GlobalInc", "DataSys"]),
            "{case_id}": f"CASE-{self.rng.randint(1000, 9999)}",
            "{employee_id}": f"EMP-{self.rng.randint(1000, 9999)}",
            "{month}": f"{self.rng.randint(1, 12):02d}",
            "{branch}": self.rng.choice(["main", "develop", "feature-x", "hotfix-y"]),
            "{environment}": self.rng.choice(["prod", "staging", "dev"]),
            "{component}": self.rng.choice(["api", "database", "frontend", "cache"]),
            "{pr_id}": str(self.rng.randint(100, 9999)),
            "{topic}": self.rng.choice(["ML", "distributed-systems", "security", "cloud"]),
            "{research_id}": f"RSH-{self.rng.randint(1000, 9999)}",
            "{grant_id}": f"GNT-{self.rng.randint(1000, 9999)}",
            "{study_id}": f"STD-{self.rng.randint(1000, 9999)}",
            "{platform}": self.rng.choice(["twitter", "linkedin", "facebook", "instagram"]),
            "{template_id}": f"TPL-{self.rng.randint(1000, 9999)}",
            "{campaign_id}": f"CMP-{self.rng.randint(1000, 9999)}",
            "{location}": self.rng.choice(["warehouse-A", "warehouse-B", "office-HQ"]),
            "{vendor_id}": f"VND-{self.rng.randint(100, 999)}",
            "{facility_id}": f"FAC-{self.rng.randint(100, 999)}",
            "{process_id}": f"PRC-{self.rng.randint(1000, 9999)}",
            "{incident_id}": f"INC-{self.rng.randint(1000, 9999)}",
            "{scan_id}": f"SCN-{self.rng.randint(1000, 9999)}",
            "{audit_id}": f"AUD-{self.rng.randint(1000, 9999)}",
            "{host}": self.rng.choice(["server-01", "server-02", "db-01", "db-02"]),
            "{backup_id}": f"BKP-{self.rng.randint(10000, 99999)}",
            "{standard}": self.rng.choice(["SOC2", "ISO27001", "HIPAA", "PCI-DSS"]),
        }
        
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, str(value))
        
        # Add random extension
        ext = self.rng.choice([".pdf", ".docx", ".xlsx", ".txt", ".json", ".csv"])
        return template + ext
    
    def _generate_content(self, category: Category, index: int) -> str:
        """Generate realistic content snippet for category."""
        templates = self._CATEGORY_TEMPLATES[category]["content_samples"]
        template = self.rng.choice(templates)
        
        # Fill template with random values
        def format_float():
            return round(self.rng.uniform(0.1, 100.0), 2)
        
        replacements = {
            "{}": [str(self.rng.randint(1, 100))],
            "{:.2f}": [str(format_float())],
            "{:.1f}": [str(round(self.rng.uniform(10.0, 1000.0), 1))],
            "{:.0%}": [f"{self.rng.randint(50, 99)}%"],
            "{:.0}%": [str(self.rng.randint(50, 99))],
            "{}%": [str(self.rng.randint(50, 99))],
        }
        
        content = template
        for placeholder, values in replacements.items():
            for value in values:
                content = content.replace(placeholder, str(value), 1)
        
        # Clean up any remaining placeholders
        while "{" in content:
            content = content.replace("{}", str(self.rng.randint(1, 100)), 1)
            content = content.replace("{", "", 1)
            content = content.replace("}", "", 1)
        
        return content
    
    def _generate_size(self, category: Category) -> int:
        """Generate realistic file size for category."""
        min_size, max_size = self._CATEGORY_TEMPLATES[category]["size_range"]
        # Use log-uniform distribution for more realistic sizes
        log_min = __import__("math").log(min_size)
        log_max = __import__("math").log(max_size)
        log_size = self.rng.uniform(log_min, log_max)
        size = int(__import__("math").exp(log_size))
        return min(max(size, min_size), max_size)
    
    def _generate_metadata(self, category: Category, index: int) -> Dict:
        """Generate metadata for a file."""
        return {
            "category": category.value,
            "file_type": self.rng.choice(["document", "archive", "database", "log"]),
            "department": self.rng.choice(["IT", "Finance", "HR", "Operations"]),
            "confidentiality": self.rng.choice(["public", "internal", "confidential", "restricted"]),
            "indexed": self.rng.choice([True, False]),
            "compressed": self.rng.choice([True, False]),
            "checksum": f"sha256_{self.rng.randint(100000, 999999):06x}",
        }
    
    def to_jsonl(self, path: Path) -> None:
        """
        Write generated files to JSON Lines format with schema validation.
        
        Args:
            path: Path to write the JSONL file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            for file_doc in self.files:
                # Validate the document
                validated = FileDocument(**file_doc.model_dump())
                json_line = validated.model_dump_json()
                f.write(json_line + "\n")
