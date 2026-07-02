"""
Unit tests for custom exception hierarchy.

Tests the ExplanationStudyError base exception and all specific exception types
(DataLoadError, APIError, ParsingError, ValidationError, ConfigurationError).

Requirements: 20.2, 20.3
"""

import pytest
from src.utils.exceptions import (
    ExplanationStudyError,
    DataLoadError,
    APIError,
    ParsingError,
    ValidationError,
    ConfigurationError,
    raise_data_load_error,
    raise_api_error,
    raise_parsing_error,
    raise_validation_error,
    raise_configuration_error,
)


class TestExplanationStudyError:
    """Test base exception class."""
    
    def test_base_exception_with_defaults(self):
        """Test base exception with default parameters."""
        exc = ExplanationStudyError("Test error message")
        
        assert exc.message == "Test error message"
        assert exc.error_code == "ESE000"
        assert exc.details == {}
        assert str(exc) == "[ESE000] Test error message"
    
    def test_base_exception_with_custom_code(self):
        """Test base exception with custom error code."""
        exc = ExplanationStudyError("Test error", error_code="ESE123")
        
        assert exc.error_code == "ESE123"
        assert str(exc) == "[ESE123] Test error"
    
    def test_base_exception_with_details(self):
        """Test base exception with details dictionary."""
        details = {"dataset": "sst2", "sample_size": 200}
        exc = ExplanationStudyError("Test error", error_code="ESE001", details=details)
        
        assert exc.details == details
        assert "dataset=sst2" in str(exc)
        assert "sample_size=200" in str(exc)
    
    def test_to_dict_serialization(self):
        """Test exception serialization to dictionary."""
        details = {"model": "llama3", "strategy": "highlighting"}
        exc = ExplanationStudyError("Test error", error_code="ESE001", details=details)
        
        result = exc.to_dict()
        
        assert result["error_type"] == "ExplanationStudyError"
        assert result["error_code"] == "ESE001"
        assert result["message"] == "Test error"
        assert result["details"] == details


class TestDataLoadError:
    """Test DataLoadError exception."""
    
    def test_data_load_error_basic(self):
        """Test DataLoadError with basic parameters."""
        exc = DataLoadError("Dataset not found", error_code="DLE001")
        
        assert exc.message == "Dataset not found"
        assert exc.error_code == "DLE001"
        assert isinstance(exc, ExplanationStudyError)
    
    def test_data_load_error_with_details(self):
        """Test DataLoadError with context details."""
        details = {"dataset_name": "mnli", "split": "train"}
        exc = DataLoadError("Sampling failed", error_code="DLE003", details=details)
        
        assert exc.details == details
        assert "dataset_name=mnli" in str(exc)
    
    def test_raise_data_load_error_convenience(self):
        """Test convenience function for raising DataLoadError."""
        with pytest.raises(DataLoadError) as exc_info:
            raise_data_load_error(
                "Export failed",
                error_code="DLE005",
                output_path="/data/output.json"
            )
        
        exc = exc_info.value
        assert exc.error_code == "DLE005"
        assert exc.details["output_path"] == "/data/output.json"


class TestAPIError:
    """Test APIError exception."""
    
    def test_api_error_basic(self):
        """Test APIError with basic parameters."""
        exc = APIError("Authentication failed", error_code="API001")
        
        assert exc.message == "Authentication failed"
        assert exc.error_code == "API001"
        assert isinstance(exc, ExplanationStudyError)
    
    def test_api_error_with_retry_context(self):
        """Test APIError with retry details."""
        details = {"model_name": "llama3", "retry_count": 3, "status_code": 429}
        exc = APIError("Rate limit exceeded", error_code="API003", details=details)
        
        assert exc.details["retry_count"] == 3
        assert exc.details["status_code"] == 429
    
    def test_raise_api_error_convenience(self):
        """Test convenience function for raising APIError."""
        with pytest.raises(APIError) as exc_info:
            raise_api_error(
                "Request timeout",
                error_code="API002",
                model_name="llama3",
                timeout_seconds=30
            )
        
        exc = exc_info.value
        assert exc.error_code == "API002"
        assert exc.details["timeout_seconds"] == 30


class TestParsingError:
    """Test ParsingError exception."""
    
    def test_parsing_error_basic(self):
        """Test ParsingError with basic parameters."""
        exc = ParsingError("Classification parsing failed", error_code="PRS001")
        
        assert exc.message == "Classification parsing failed"
        assert exc.error_code == "PRS001"
        assert isinstance(exc, ExplanationStudyError)
    
    def test_parsing_error_with_raw_response(self):
        """Test ParsingError with raw model response."""
        details = {
            "strategy": "highlighting",
            "raw_response": "Token1, Token2",
            "instance_id": "sst2_001"
        }
        exc = ParsingError("Incomplete extraction", error_code="PRS009", details=details)
        
        assert exc.details["strategy"] == "highlighting"
        assert "raw_response" in exc.details
    
    def test_raise_parsing_error_convenience(self):
        """Test convenience function for raising ParsingError."""
        with pytest.raises(ParsingError) as exc_info:
            raise_parsing_error(
                "Fuzzy matching failed",
                error_code="PRS007",
                pattern="confidence: (\\d+)",
                text="invalid response"
            )
        
        exc = exc_info.value
        assert exc.error_code == "PRS007"
        assert "pattern" in exc.details


class TestValidationError:
    """Test ValidationError exception."""
    
    def test_validation_error_basic(self):
        """Test ValidationError with basic parameters."""
        exc = ValidationError("Missing required parameter", error_code="VAL001")
        
        assert exc.message == "Missing required parameter"
        assert exc.error_code == "VAL001"
        assert isinstance(exc, ExplanationStudyError)
    
    def test_validation_error_with_expected_actual(self):
        """Test ValidationError with expected and actual values."""
        details = {
            "parameter_name": "temperature",
            "expected_value": 0,
            "actual_value": 0.7
        }
        exc = ValidationError("Invalid configuration", error_code="VAL002", details=details)
        
        assert exc.details["expected_value"] == 0
        assert exc.details["actual_value"] == 0.7
    
    def test_raise_validation_error_convenience(self):
        """Test convenience function for raising ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            raise_validation_error(
                "Checkpoint corrupted",
                error_code="VAL003",
                checkpoint_file="/outputs/checkpoint.json"
            )
        
        exc = exc_info.value
        assert exc.error_code == "VAL003"


class TestConfigurationError:
    """Test ConfigurationError exception."""
    
    def test_configuration_error_basic(self):
        """Test ConfigurationError with basic parameters."""
        exc = ConfigurationError("Config file not found", error_code="CFG001")
        
        assert exc.message == "Config file not found"
        assert exc.error_code == "CFG001"
        assert isinstance(exc, ExplanationStudyError)
    
    def test_configuration_error_missing_env_var(self):
        """Test ConfigurationError for missing environment variable."""
        details = {"env_var": "AWS_ACCESS_KEY_ID"}
        exc = ConfigurationError(
            "Missing required environment variable",
            error_code="CFG003",
            details=details
        )

        assert exc.details["env_var"] == "AWS_ACCESS_KEY_ID"
    
    def test_raise_configuration_error_convenience(self):
        """Test convenience function for raising ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise_configuration_error(
                "Invalid YAML syntax",
                error_code="CFG002",
                config_file="/config/experiment.yaml",
                line_number=42
            )
        
        exc = exc_info.value
        assert exc.error_code == "CFG002"
        assert exc.details["config_file"] == "/config/experiment.yaml"


class TestExceptionHierarchy:
    """Test exception hierarchy relationships."""
    
    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from ExplanationStudyError."""
        exception_types = [
            DataLoadError,
            APIError,
            ParsingError,
            ValidationError,
            ConfigurationError,
        ]
        
        for exc_type in exception_types:
            exc = exc_type("Test message")
            assert isinstance(exc, ExplanationStudyError)
            assert isinstance(exc, Exception)
    
    def test_catching_base_exception_catches_all(self):
        """Test that catching base exception catches all specific types."""
        with pytest.raises(ExplanationStudyError):
            raise DataLoadError("Test")
        
        with pytest.raises(ExplanationStudyError):
            raise APIError("Test")
        
        with pytest.raises(ExplanationStudyError):
            raise ParsingError("Test")
        
        with pytest.raises(ExplanationStudyError):
            raise ValidationError("Test")
        
        with pytest.raises(ExplanationStudyError):
            raise ConfigurationError("Test")
    
    def test_specific_exception_catching(self):
        """Test that specific exceptions can be caught independently."""
        with pytest.raises(DataLoadError):
            raise DataLoadError("Dataset error")
        
        with pytest.raises(APIError):
            raise APIError("API error")
        
        # Verify that catching specific type doesn't catch others
        with pytest.raises(DataLoadError):
            try:
                raise DataLoadError("Test")
            except APIError:
                pytest.fail("Should not catch APIError")


class TestErrorCodes:
    """Test error code conventions and structure."""
    
    def test_error_codes_follow_convention(self):
        """Test that error codes follow the expected format."""
        test_cases = [
            (DataLoadError("Test", "DLE001"), "DLE001"),
            (APIError("Test", "API003"), "API003"),
            (ParsingError("Test", "PRS007"), "PRS007"),
            (ValidationError("Test", "VAL002"), "VAL002"),
            (ConfigurationError("Test", "CFG005"), "CFG005"),
        ]
        
        for exc, expected_code in test_cases:
            assert exc.error_code == expected_code
            assert expected_code in str(exc)
    
    def test_default_error_codes(self):
        """Test that default error codes are assigned correctly."""
        assert DataLoadError("Test").error_code == "DLE000"
        assert APIError("Test").error_code == "API000"
        assert ParsingError("Test").error_code == "PRS000"
        assert ValidationError("Test").error_code == "VAL000"
        assert ConfigurationError("Test").error_code == "CFG000"


class TestExceptionSerialization:
    """Test exception serialization for logging."""
    
    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all required fields."""
        exc = ParsingError(
            "Extraction failed",
            error_code="PRS003",
            details={"strategy": "highlighting"}
        )
        
        result = exc.to_dict()
        
        assert "error_type" in result
        assert "error_code" in result
        assert "message" in result
        assert "details" in result
    
    def test_to_dict_preserves_exception_type(self):
        """Test that serialized dict preserves the specific exception type."""
        exceptions = [
            DataLoadError("Test", "DLE001"),
            APIError("Test", "API001"),
            ParsingError("Test", "PRS001"),
            ValidationError("Test", "VAL001"),
            ConfigurationError("Test", "CFG001"),
        ]
        
        expected_types = [
            "DataLoadError",
            "APIError",
            "ParsingError",
            "ValidationError",
            "ConfigurationError",
        ]
        
        for exc, expected_type in zip(exceptions, expected_types):
            result = exc.to_dict()
            assert result["error_type"] == expected_type
    
    def test_to_dict_handles_empty_details(self):
        """Test that to_dict handles exceptions with no details."""
        exc = APIError("Test error", "API001")
        result = exc.to_dict()
        
        assert result["details"] == {}
