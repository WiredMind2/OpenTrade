"""
Data validation and quality monitoring for trading backtesting system.

This module provides comprehensive data validation, quality checks, anomaly detection,
and data quality monitoring for financial data, news articles, and model predictions.
"""
import pandas as pd
import numpy as np
import sqlite3
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
import json
from pathlib import Path

from backend.config import get_config
from backend.logging_config import get_component_logger
from backend.error_handling import handle_data_errors, DataIngestionError


logger = get_component_logger(__file__)


class DataQualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class ValidationRule(Enum):
    """Data validation rule types."""
    RANGE_CHECK = "range_check"
    NULL_CHECK = "null_check"
    DUPLICATE_CHECK = "duplicate_check"
    FORMAT_CHECK = "format_check"
    BUSINESS_RULE = "business_rule"
    STATISTICAL_CHECK = "statistical_check"
    REFERENTIAL_INTEGRITY = "referential_integrity"


@dataclass
class ValidationResult:
    """Result of a data validation check."""
    rule: ValidationRule
    table: str
    column: Optional[str] = None
    passed: bool = True
    message: str = ""
    severity: str = "info"
    affected_rows: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    timestamp: datetime
    table_name: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    quality_score: float
    quality_level: DataQualityLevel
    validation_results: List[ValidationResult]
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]


class DataValidator:
    """Core data validation engine."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.validation_rules = {}
        self.thresholds = {
            'null_threshold': 0.05,  # 5% null values allowed
            'duplicate_threshold': 0.01,  # 1% duplicates allowed
            'outlier_threshold': 0.03,  # 3% outliers allowed
            'quality_score_threshold': 0.85  # 85% quality score required
        }
    
    @handle_data_errors
    def validate_table(self, table_name: str, rules: List[Dict[str, Any]] | None = None) -> DataQualityReport:
        """Validate a database table."""
        logger.info(f"Starting validation for table: {table_name}")
        
        # Get table data
        df = self._load_table_data(table_name)
        if df.empty:
            raise DataIngestionError(f"Table {table_name} is empty or doesn't exist")
        
        # Apply validation rules
        validation_results = []
        if rules:
            for rule_config in rules:
                result = self._apply_validation_rule(df, table_name, rule_config)
                validation_results.append(result)
        else:
            # Apply default validation rules
            validation_results = self._apply_default_rules(df, table_name)
        
        # Calculate quality metrics
        valid_rows = sum(1 for r in validation_results if r.passed)
        total_checks = len(validation_results)
        quality_score = valid_rows / total_checks if total_checks > 0 else 1.0
        
        # Determine quality level
        if quality_score >= 0.95:
            quality_level = DataQualityLevel.EXCELLENT
        elif quality_score >= 0.85:
            quality_level = DataQualityLevel.GOOD
        elif quality_score >= 0.70:
            quality_level = DataQualityLevel.FAIR
        elif quality_score >= 0.50:
            quality_level = DataQualityLevel.POOR
        else:
            quality_level = DataQualityLevel.CRITICAL
        
        # Detect anomalies
        anomalies = self._detect_anomalies(df, table_name)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results, anomalies)
        
        report = DataQualityReport(
            timestamp=datetime.utcnow(),
            table_name=table_name,
            total_rows=len(df),
            valid_rows=valid_rows,
            invalid_rows=total_checks - valid_rows,
            quality_score=quality_score,
            quality_level=quality_level,
            validation_results=validation_results,
            anomalies=anomalies,
            recommendations=recommendations
        )
        
        # Log quality report
        logger.info(
            f"Validation complete for {table_name}: {quality_level.value} "
            f"(score: {quality_score:.2%})"
        )
        
        return report
    
    def _load_table_data(self, table_name: str) -> pd.DataFrame:
        """Load table data into DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    
    def _apply_default_rules(self, df: pd.DataFrame, table_name: str) -> List[ValidationResult]:
        """Apply default validation rules based on table structure."""
        results = []
        
        for column in df.columns:
            # Null value check
            null_result = self._check_null_values(df, table_name, column)
            results.append(null_result)
            
            # Data type validation based on column name patterns
            if 'date' in column.lower() or 'time' in column.lower():
                date_result = self._check_date_format(df, table_name, column)
                results.append(date_result)
            
            if 'price' in column.lower() or 'return' in column.lower():
                numeric_result = self._check_numeric_values(df, table_name, column)
                results.append(numeric_result)
            
            if 'ticker' in column.lower():
                ticker_result = self._check_ticker_format(df, table_name, column)
                results.append(ticker_result)
        
        # Check for duplicates if table has appropriate columns
        duplicate_result = self._check_duplicates(df, table_name)
        results.append(duplicate_result)
        
        return results
    
    def _apply_validation_rule(self, df: pd.DataFrame, table_name: str, rule_config: Dict[str, Any]) -> ValidationResult:
        """Apply a specific validation rule."""
        rule_type = ValidationRule(rule_config['type'])
        column = rule_config.get('column')
        
        if rule_type == ValidationRule.RANGE_CHECK:
            return self._check_range(df, table_name, column, rule_config)
        elif rule_type == ValidationRule.NULL_CHECK:
            return self._check_null_values(df, table_name, column)
        elif rule_type == ValidationRule.DUPLICATE_CHECK:
            return self._check_duplicates(df, table_name, rule_config.get('columns'))
        elif rule_type == ValidationRule.FORMAT_CHECK:
            return self._check_format(df, table_name, column, rule_config)
        elif rule_type == ValidationRule.BUSINESS_RULE:
            return self._check_business_rule(df, table_name, rule_config)
        elif rule_type == ValidationRule.STATISTICAL_CHECK:
            return self._check_statistical_rule(df, table_name, column, rule_config)
        else:
            return ValidationResult(
                rule=rule_type,
                table=table_name,
                column=column,
                passed=False,
                message=f"Unknown validation rule: {rule_type}"
            )
    
    def _check_null_values(self, df: pd.DataFrame, table_name: str, column: str) -> ValidationResult:
        """Check for null values in a column."""
        null_count = df[column].isnull().sum()
        null_percentage = null_count / len(df)
        
        passed = null_percentage <= self.thresholds['null_threshold']
        severity = "error" if null_percentage > 0.1 else "warning" if null_percentage > 0.05 else "info"
        
        return ValidationResult(
            rule=ValidationRule.NULL_CHECK,
            table=table_name,
            column=column,
            passed=passed,
            message=f"Null values: {null_count} ({null_percentage:.2%})",
            severity=severity,
            affected_rows=null_count,
            details={
                "null_count": null_count,
                "null_percentage": null_percentage,
                "threshold": self.thresholds['null_threshold']
            }
        )
    
    def _check_duplicates(self, df: pd.DataFrame, table_name: str, columns: List[str] | None = None) -> ValidationResult:
        """Check for duplicate rows."""
        if columns:
            # Check duplicates on specific columns
            subset_df = df[columns]
            duplicates = subset_df.duplicated().sum()
        else:
            # Check full row duplicates
            duplicates = df.duplicated().sum()
        
        duplicate_percentage = duplicates / len(df)
        passed = duplicate_percentage <= self.thresholds['duplicate_threshold']
        severity = "error" if duplicate_percentage > 0.05 else "warning"
        
        return ValidationResult(
            rule=ValidationRule.DUPLICATE_CHECK,
            table=table_name,
            passed=passed,
            message=f"Duplicate rows: {duplicates} ({duplicate_percentage:.2%})",
            severity=severity,
            affected_rows=duplicates,
            details={
                "duplicate_count": duplicates,
                "duplicate_percentage": duplicate_percentage,
                "columns_checked": columns or "all_columns"
            }
        )
    
    def _check_date_format(self, df: pd.DataFrame, table_name: str, column: str) -> ValidationResult:
        """Check date format validation."""
        # Try to parse dates
        date_issues = 0
        total_dates = 0
        
        for idx, value in df[column].items():
            if pd.notna(value):
                total_dates += 1
                try:
                    pd.to_datetime(str(value))
                except (ValueError, TypeError):
                    date_issues += 1
        
        if total_dates == 0:
            return ValidationResult(
                rule=ValidationRule.FORMAT_CHECK,
                table=table_name,
                column=column,
                passed=True,
                message="No date values to validate"
            )
        
        issue_percentage = date_issues / total_dates
        passed = issue_percentage <= 0.01  # 1% date format errors allowed
        
        return ValidationResult(
            rule=ValidationRule.FORMAT_CHECK,
            table=table_name,
            column=column,
            passed=passed,
            message=f"Date format errors: {date_issues}/{total_dates} ({issue_percentage:.2%})",
            severity="error" if issue_percentage > 0.05 else "warning",
            affected_rows=date_issues,
            details={
                "error_count": date_issues,
                "total_count": total_dates,
                "error_percentage": issue_percentage
            }
        )
    
    def _check_numeric_values(self, df: pd.DataFrame, table_name: str, column: str) -> ValidationResult:
        """Check numeric value validation."""
        numeric_values = pd.to_numeric(df[column], errors='coerce')
        non_numeric_count = numeric_values.isnull().sum() - df[column].isnull().sum()
        total_non_null = df[column].notna().sum()
        
        if total_non_null == 0:
            return ValidationResult(
                rule=ValidationRule.FORMAT_CHECK,
                table=table_name,
                column=column,
                passed=True,
                message="No numeric values to validate"
            )
        
        error_percentage = non_numeric_count / total_non_null
        passed = error_percentage <= 0.01  # 1% numeric errors allowed
        
        # Check for reasonable financial values
        if column.lower() in ['price', 'close', 'open', 'high', 'low']:
            negative_prices = (numeric_values < 0).sum()
            extremely_high = (numeric_values > 10000).sum()  # > $10,000 per share
            
            if negative_prices > 0:
                return ValidationResult(
                    rule=ValidationRule.BUSINESS_RULE,
                    table=table_name,
                    column=column,
                    passed=False,
                    message=f"Found {negative_prices} negative prices",
                    severity="error",
                    affected_rows=negative_prices,
                    details={"negative_prices": negative_prices}
                )
        
        return ValidationResult(
            rule=ValidationRule.FORMAT_CHECK,
            table=table_name,
            column=column,
            passed=passed,
            message=f"Numeric format errors: {non_numeric_count}/{total_non_null} ({error_percentage:.2%})",
            severity="error" if error_percentage > 0.05 else "warning",
            affected_rows=non_numeric_count,
            details={
                "error_count": non_numeric_count,
                "total_count": total_non_null,
                "error_percentage": error_percentage
            }
        )
    
    def _check_ticker_format(self, df: pd.DataFrame, table_name: str, column: str) -> ValidationResult:
        """Check ticker symbol format."""
        # Tickers should be 1-5 uppercase letters
        ticker_pattern = df[column].str.match(r'^[A-Z]{1,5}$', na=False)
        invalid_tickers = (~ticker_pattern & df[column].notna()).sum()
        total_tickers = df[column].notna().sum()
        
        if total_tickers == 0:
            return ValidationResult(
                rule=ValidationRule.FORMAT_CHECK,
                table=table_name,
                column=column,
                passed=True,
                message="No ticker values to validate"
            )
        
        error_percentage = invalid_tickers / total_tickers
        passed = error_percentage <= 0.01  # 1% format errors allowed
        
        return ValidationResult(
            rule=ValidationRule.FORMAT_CHECK,
            table=table_name,
            column=column,
            passed=passed,
            message=f"Ticker format errors: {invalid_tickers}/{total_tickers} ({error_percentage:.2%})",
            severity="warning",
            affected_rows=invalid_tickers,
            details={
                "error_count": invalid_tickers,
                "total_count": total_tickers,
                "error_percentage": error_percentage
            }
        )
    
    def _check_range(self, df: pd.DataFrame, table_name: str, column: str, rule_config: Dict[str, Any]) -> ValidationResult:
        """Check if values are within specified range."""
        min_val = rule_config.get('min')
        max_val = rule_config.get('max')
        
        if min_val is not None:
            below_min = (df[column] < min_val).sum()
        else:
            below_min = 0
        
        if max_val is not None:
            above_max = (df[column] > max_val).sum()
        else:
            above_max = 0
        
        out_of_range = below_min + above_max
        total_values = df[column].notna().sum()
        
        if total_values == 0:
            return ValidationResult(
                rule=ValidationRule.RANGE_CHECK,
                table=table_name,
                column=column,
                passed=True,
                message="No values to check range"
            )
        
        error_percentage = out_of_range / total_values
        passed = error_percentage <= rule_config.get('threshold', 0.05)
        
        return ValidationResult(
            rule=ValidationRule.RANGE_CHECK,
            table=table_name,
            column=column,
            passed=passed,
            message=f"Out of range values: {out_of_range}/{total_values} ({error_percentage:.2%})",
            severity="error" if error_percentage > 0.1 else "warning",
            affected_rows=out_of_range,
            details={
                "below_min": below_min,
                "above_max": above_max,
                "total_out_of_range": out_of_range,
                "total_values": total_values,
                "min_allowed": min_val,
                "max_allowed": max_val
            }
        )
    
    def _check_business_rule(self, df: pd.DataFrame, table_name: str, rule_config: Dict[str, Any]) -> ValidationResult:
        """Check business rule compliance."""
        rule_name = rule_config['name']
        
        # Implement specific business rules
        if rule_name == "price_high_not_less_than_low":
            # High price should not be less than low price
            if 'high' in df.columns and 'low' in df.columns:
                invalid_prices = (df['high'] < df['low']).sum()
                total_prices = df[['high', 'low']].notna().all(axis=1).sum()
                
                if total_prices > 0:
                    error_percentage = invalid_prices / total_prices
                    passed = error_percentage <= 0.01
                    
                    return ValidationResult(
                        rule=ValidationRule.BUSINESS_RULE,
                        table=table_name,
                        passed=passed,
                        message=f"High < Low violations: {invalid_prices}/{total_prices} ({error_percentage:.2%})",
                        severity="error",
                        affected_rows=invalid_prices,
                        details={"rule": rule_name}
                    )
        
        # Default business rule result
        return ValidationResult(
            rule=ValidationRule.BUSINESS_RULE,
            table=table_name,
            passed=True,
            message=f"Business rule '{rule_name}' passed"
        )
    
    def _check_statistical_rule(self, df: pd.DataFrame, table_name: str, column: str, rule_config: Dict[str, Any]) -> ValidationResult:
        """Check statistical rules (outliers, etc.)."""
        # Z-score based outlier detection
        if df[column].dtype in ['float64', 'int64']:
            values = df[column].dropna()
            if len(values) > 0:
                z_scores = np.abs((values - values.mean()) / values.std())
                outliers = (z_scores > 3).sum()
                outlier_percentage = outliers / len(values)
                
                passed = outlier_percentage <= self.thresholds['outlier_threshold']
                
                return ValidationResult(
                    rule=ValidationRule.STATISTICAL_CHECK,
                    table=table_name,
                    column=column,
                    passed=passed,
                    message=f"Statistical outliers: {outliers}/{len(values)} ({outlier_percentage:.2%})",
                    severity="info",
                    affected_rows=outliers,
                    details={
                        "outlier_count": outliers,
                        "total_count": len(values),
                        "outlier_percentage": outlier_percentage,
                        "z_score_threshold": 3
                    }
                )
        
        return ValidationResult(
            rule=ValidationRule.STATISTICAL_CHECK,
            table=table_name,
            column=column,
            passed=True,
            message="No numeric data for statistical check"
        )
    
    def _check_format(self, df: pd.DataFrame, table_name: str, column: str, rule_config: Dict[str, Any]) -> ValidationResult:
        """Check format compliance."""
        pattern = rule_config.get('pattern')
        
        if pattern:
            import re
            valid_format = df[column].str.match(pattern, na=False)
            invalid_format = (~valid_format & df[column].notna()).sum()
            total_values = df[column].notna().sum()
            
            if total_values > 0:
                error_percentage = invalid_format / total_values
                passed = error_percentage <= 0.01
                
                return ValidationResult(
                    rule=ValidationRule.FORMAT_CHECK,
                    table=table_name,
                    column=column,
                    passed=passed,
                    message=f"Format violations: {invalid_format}/{total_values} ({error_percentage:.2%})",
                    severity="warning",
                    affected_rows=invalid_format,
                    details={
                        "error_count": invalid_format,
                        "total_count": total_values,
                        "pattern": pattern
                    }
                )
        
        return ValidationResult(
            rule=ValidationRule.FORMAT_CHECK,
            table=table_name,
            column=column,
            passed=True,
            message="Format check passed"
        )
    
    def _detect_anomalies(self, df: pd.DataFrame, table_name: str) -> List[Dict[str, Any]]:
        """Detect data anomalies using various methods."""
        anomalies = []
        
        # Detect sudden spikes/drops in time series data
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64'] and 'date' not in column.lower():
                # Z-score anomaly detection
                values = df[column].dropna()
                if len(values) > 10:  # Need sufficient data
                    z_scores = np.abs((values - values.mean()) / values.std())
                    anomalous_indices = df[z_scores > 3].index.tolist()
                    
                    if anomalous_indices:
                        anomalies.append({
                            "type": "statistical_outlier",
                            "column": column,
                            "count": len(anomalous_indices),
                            "indices": anomalous_indices[:10],  # First 10 for reference
                            "description": f"Found {len(anomalous_indices)} outliers in {column}"
                        })
        
        # Check for sudden changes in data volume
        if 'date' in df.columns:
            try:
                df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
                daily_counts = df.groupby(df['date_parsed'].dt.date).size()
                
                if len(daily_counts) > 7:  # Need at least a week of data
                    mean_count = daily_counts.mean()
                    std_count = daily_counts.std()
                    
                    # Find days with unusual volume
                    unusual_days = daily_counts[daily_counts > mean_count + 2 * std_count]
                    if len(unusual_days) > 0:
                        anomalies.append({
                            "type": "volume_spike",
                            "description": f"Found {len(unusual_days)} days with unusual data volume",
                            "details": unusual_days.to_dict()
                        })
            except Exception as e:
                logger.warning(f"Could not detect volume anomalies: {e}")
        
        return anomalies
    
    def validate_symbol_exists(self, symbol: str) -> bool:
        """Check if a symbol exists in the tickers table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM tickers WHERE ticker = ? LIMIT 1", (symbol.upper(),))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking if symbol {symbol} exists: {e}")
            return False

    def _generate_recommendations(self, validation_results: List[ValidationResult], anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Analyze validation results
        failed_results = [r for r in validation_results if not r.passed]

        if failed_results:
            recommendations.append("Address failed validation rules:")

            # Group by severity
            errors = [r for r in failed_results if r.severity == "error"]
            warnings = [r for r in failed_results if r.severity == "warning"]

            if errors:
                recommendations.append(f"  - Fix {len(errors)} critical data quality issues")

            if warnings:
                recommendations.append(f"  - Address {len(warnings)} data quality warnings")

            # Specific recommendations
            null_checks = [r for r in failed_results if r.rule == ValidationRule.NULL_CHECK]
            if null_checks:
                recommendations.append("  - Implement data imputation strategies for null values")

            duplicate_checks = [r for r in failed_results if r.rule == ValidationRule.DUPLICATE_CHECK]
            if duplicate_checks:
                recommendations.append("  - Review data deduplication procedures")

            format_checks = [r for r in failed_results if r.rule == ValidationRule.FORMAT_CHECK]
            if format_checks:
                recommendations.append("  - Strengthen input validation at data ingestion")

        # Analyze anomalies
        if anomalies:
            recommendations.append("Investigate detected anomalies:")

            outlier_count = sum(1 for a in anomalies if a["type"] == "statistical_outlier")
            if outlier_count > 0:
                recommendations.append(f"  - Review {outlier_count} statistical outlier cases")

            volume_spikes = [a for a in anomalies if a["type"] == "volume_spike"]
            if volume_spikes:
                recommendations.append("  - Investigate unusual data volume patterns")

        # General recommendations
        if not recommendations:
            recommendations.append("Data quality is good. Continue monitoring.")

        return recommendations


class DataQualityMonitor:
    """Monitor data quality over time and alert on degradation."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.validator = DataValidator(db_path)
        self.quality_history = []
    
    def run_quality_checks(self, tables: List[str] = None) -> Dict[str, DataQualityReport]:
        """Run quality checks on specified tables."""
        if tables is None:
            # Default tables to check
            tables = ['tickers', 'articles', 'price_daily', 'sentiment_predictions']
        
        reports = {}
        
        for table_name in tables:
            try:
                report = self.validator.validate_table(table_name)
                reports[table_name] = report
                
                # Store in quality history
                self.quality_history.append({
                    'timestamp': report.timestamp,
                    'table': table_name,
                    'quality_score': report.quality_score,
                    'quality_level': report.quality_level.value,
                    'total_issues': len([r for r in report.validation_results if not r.passed])
                })
                
                # Alert if quality is poor
                if report.quality_level in [DataQualityLevel.POOR, DataQualityLevel.CRITICAL]:
                    self._send_quality_alert(table_name, report)
                
            except Exception as e:
                logger.error(f"Failed to validate table {table_name}: {e}")
                # Create error report
                error_report = DataQualityReport(
                    timestamp=datetime.utcnow(),
                    table_name=table_name,
                    total_rows=0,
                    valid_rows=0,
                    invalid_rows=0,
                    quality_score=0.0,
                    quality_level=DataQualityLevel.CRITICAL,
                    validation_results=[],
                    anomalies=[],
                    recommendations=[f"Validation failed: {str(e)}"]
                )
                reports[table_name] = error_report
        
        return reports
    
    def _send_quality_alert(self, table_name: str, report: DataQualityReport):
        """Send alert for poor data quality."""
        alert_message = (
            f"ALERT: Poor data quality detected in table '{table_name}'. "
            f"Quality score: {report.quality_score:.2%}, "
            f"Level: {report.quality_level.value}. "
            f"Issues found: {len([r for r in report.validation_results if not r.passed])}"
        )
        
        logger.critical(alert_message, extra={
            "alert_type": "data_quality",
            "table": table_name,
            "quality_score": report.quality_score,
            "quality_level": report.quality_level.value
        })
        
        # In production, would send to monitoring/alerting system
        # send_to_alerting_system(alert_message)
    
    def get_quality_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get data quality trends over time."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_history = [
            record for record in self.quality_history
            if record['timestamp'] >= cutoff_date
        ]
        
        if not recent_history:
            return {"message": "No quality history available"}
        
        # Group by table
        trends = {}
        tables = set(record['table'] for record in recent_history)
        
        for table in tables:
            table_records = [r for r in recent_history if r['table'] == table]
            table_records.sort(key=lambda x: x['timestamp'])
            
            scores = [r['quality_score'] for r in table_records]
            trend_data = {
                "table": table,
                "average_score": np.mean(scores),
                "latest_score": scores[-1] if scores else 0,
                "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "declining" if len(scores) > 1 else "stable",
                "check_count": len(table_records),
                "issues_detected": sum(r['total_issues'] for r in table_records)
            }
            
            trends[table] = trend_data
        
        return trends


def create_data_quality_monitor(db_path: str = None) -> DataQualityMonitor:
    """Create and initialize data quality monitor."""
    if db_path is None:
        config = get_config()
        db_path = config.database.path

    return DataQualityMonitor(db_path)


def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
    """Validate that start date is before end date.

    Args:
        start_date: Start date as datetime object
        end_date: End date as datetime object

    Returns:
        True if validation passes

    Raises:
        ValueError: If start date is not before end date
    """
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")
    return True


# Example usage and testing
if __name__ == "__main__":
    # Command line interface for data quality monitoring
    import argparse
    
    parser = argparse.ArgumentParser(description="Data quality monitoring")
    parser.add_argument("--db", default="data/backtest.db", help="Database path")
    parser.add_argument("--tables", nargs="+", help="Tables to validate")
    parser.add_argument("--output", help="Output file for quality report")
    
    args = parser.parse_args()
    
    monitor = create_data_quality_monitor(args.db)
    reports = monitor.run_quality_checks(args.tables)
    
    # Print summary
    for table_name, report in reports.items():
        print(f"\n{table_name}:")
        print(f"  Quality: {report.quality_level.value} ({report.quality_score:.2%})")
        print(f"  Issues: {len([r for r in report.validation_results if not r.passed])}")
        print(f"  Rows: {report.total_rows}")
        
        if report.recommendations:
            print("  Recommendations:")
            for rec in report.recommendations:
                print(f"    - {rec}")
    
    # Save detailed report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                table_name: {
                    "quality_score": report.quality_score,
                    "quality_level": report.quality_level.value,
                    "total_rows": report.total_rows,
                    "validation_results": [
                        {
                            "rule": r.rule.value,
                            "passed": r.passed,
                            "message": r.message,
                            "severity": r.severity,
                            "affected_rows": r.affected_rows
                        } for r in report.validation_results
                    ],
                    "anomalies": report.anomalies,
                    "recommendations": report.recommendations
                } for table_name, report in reports.items()
            }, f, indent=2, default=str)