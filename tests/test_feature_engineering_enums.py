"""
Unit tests for feature engineering enums and data classes.
"""
import pytest
from backend.feature_engineering import FeatureType, FeatureDefinition, FeatureSet
from datetime import datetime


@pytest.mark.unit
class TestFeatureType:
    """Test FeatureType enum."""

    def test_feature_type_values(self):
        """Test FeatureType enum values."""
        assert FeatureType.TECHNICAL.value == "technical"
        assert FeatureType.SENTIMENT.value == "sentiment"
        assert FeatureType.FUNDAMENTAL.value == "fundamental"
        assert FeatureType.BEHAVIORAL.value == "behavioral"
        assert FeatureType.LAGGED.value == "lagged"
        assert FeatureType.AGGREGATED.value == "aggregated"

    def test_feature_type_members(self):
        """Test all FeatureType members exist."""
        expected_members = {
            "TECHNICAL", "SENTIMENT", "FUNDAMENTAL",
            "BEHAVIORAL", "LAGGED", "AGGREGATED"
        }
        actual_members = {member.name for member in FeatureType}
        assert actual_members == expected_members


@pytest.mark.unit
class TestFeatureDefinition:
    """Test FeatureDefinition dataclass."""

    def test_feature_definition_creation(self):
        """Test creating a FeatureDefinition."""
        feature = FeatureDefinition(
            name="test_feature",
            feature_type=FeatureType.TECHNICAL,
            calculation_func="sma",
            parameters={"period": 20},
            dependencies=["close"],
            description="Test feature",
            frequency="daily",
            window_size=20
        )

        assert feature.name == "test_feature"
        assert feature.feature_type == FeatureType.TECHNICAL
        assert feature.calculation_func == "sma"
        assert feature.parameters == {"period": 20}
        assert feature.dependencies == ["close"]
        assert feature.description == "Test feature"
        assert feature.frequency == "daily"
        assert feature.window_size == 20

    def test_feature_definition_defaults(self):
        """Test FeatureDefinition default values."""
        feature = FeatureDefinition(
            name="minimal_feature",
            feature_type=FeatureType.TECHNICAL,
            calculation_func="rsi"
        )

        assert feature.parameters == {}
        assert feature.dependencies == []
        assert feature.description == ""
        assert feature.frequency == "daily"
        assert feature.window_size is None

    def test_feature_definition_equality(self):
        """Test FeatureDefinition equality."""
        feature1 = FeatureDefinition(
            name="test", feature_type=FeatureType.TECHNICAL, calculation_func="sma"
        )
        feature2 = FeatureDefinition(
            name="test", feature_type=FeatureType.TECHNICAL, calculation_func="sma"
        )
        feature3 = FeatureDefinition(
            name="different", feature_type=FeatureType.TECHNICAL, calculation_func="sma"
        )

        assert feature1 == feature2
        assert feature1 != feature3


@pytest.mark.unit
class TestFeatureSet:
    """Test FeatureSet dataclass."""

    def test_feature_set_creation(self):
        """Test creating a FeatureSet."""
        from datetime import datetime

        features = [
            FeatureDefinition(name="feat1", feature_type=FeatureType.TECHNICAL, calculation_func="sma"),
            FeatureDefinition(name="feat2", feature_type=FeatureType.SENTIMENT, calculation_func="sentiment")
        ]

        feature_set = FeatureSet(
            name="test_set",
            description="Test feature set",
            features=features,
            target_variable="future_return",
            horizon="1d"
        )

        assert feature_set.name == "test_set"
        assert feature_set.description == "Test feature set"
        assert len(feature_set.features) == 2
        assert feature_set.target_variable == "future_return"
        assert feature_set.horizon == "1d"
        assert isinstance(feature_set.created_at, datetime)

    def test_feature_set_defaults(self):
        """Test FeatureSet default values."""
        feature_set = FeatureSet(
            name="minimal_set",
            description="Minimal feature set",
            features=[]
        )

        assert feature_set.target_variable is None
        assert feature_set.horizon is None
        assert isinstance(feature_set.created_at, datetime)

    def test_feature_set_feature_types(self):
        """Test FeatureSet with different feature types."""
        features = [
            FeatureDefinition(name="tech", feature_type=FeatureType.TECHNICAL, calculation_func="sma"),
            FeatureDefinition(name="sent", feature_type=FeatureType.SENTIMENT, calculation_func="sentiment"),
            FeatureDefinition(name="behav", feature_type=FeatureType.BEHAVIORAL, calculation_func="momentum")
        ]

        feature_set = FeatureSet(
            name="mixed_set",
            description="Mixed feature types",
            features=features
        )

        feature_types = {f.feature_type for f in feature_set.features}
        assert FeatureType.TECHNICAL in feature_types
        assert FeatureType.SENTIMENT in feature_types
        assert FeatureType.BEHAVIORAL in feature_types