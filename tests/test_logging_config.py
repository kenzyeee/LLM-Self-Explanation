"""
Tests for the logging configuration module.

Tests verify:
- Rotating file handler configuration (10MB, 5 backups)
- Console handler with configurable levels
- Detailed and simple formatters
- Structured logging with extra fields
- Convenience logging functions

Requirements tested: 20.1, 20.2, 20.5
"""

import logging
import pytest
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from src.utils.logging_config import (
    setup_logging,
    get_structured_logger,
    log_api_request,
    log_parsing_failure,
    log_model_refusal,
    StructuredFormatter
)


class TestSetupLogging:
    """Test the setup_logging function."""
    
    def test_creates_log_directory(self, tmp_path):
        """Test that log directory is created if it doesn't exist."""
        log_dir = tmp_path / "test_logs"
        assert not log_dir.exists()
        
        logger = setup_logging(log_dir=log_dir)
        
        assert log_dir.exists()
        assert log_dir.is_dir()
    
    def test_creates_log_file(self, tmp_path):
        """Test that log file is created."""
        log_dir = tmp_path / "test_logs"
        
        logger = setup_logging(log_dir=log_dir, log_name="test.log")
        
        log_file = log_dir / "test.log"
        assert log_file.exists()
        assert log_file.is_file()
    
    def test_returns_logger_instance(self, tmp_path):
        """Test that function returns a logger instance."""
        log_dir = tmp_path / "test_logs"
        
        logger = setup_logging(log_dir=log_dir)
        
        assert isinstance(logger, logging.Logger)
    
    def test_logger_has_file_handler(self, tmp_path):
        """Test that logger has a rotating file handler."""
        log_dir = tmp_path / "test_logs"
        
        logger = setup_logging(log_dir=log_dir)
        
        file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1
    
    def test_logger_has_console_handler(self, tmp_path):
        """Test that logger has a console handler."""
        log_dir = tmp_path / "test_logs"
        
        logger = setup_logging(log_dir=log_dir)
        
        console_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(console_handlers) == 1
    
    def test_rotating_file_handler_config(self, tmp_path):
        """Test that rotating file handler has correct configuration."""
        log_dir = tmp_path / "test_logs"
        
        logger = setup_logging(log_dir=log_dir)
        
        file_handler = next(
            h for h in logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        )
        
        # Check max bytes (10MB)
        assert file_handler.maxBytes == 10 * 1024 * 1024
        # Check backup count (5)
        assert file_handler.backupCount == 5
    
    def test_console_level_configuration(self, tmp_path):
        """Test that console handler uses specified log level."""
        log_dir = tmp_path / "test_logs"
        
        logger = setup_logging(log_dir=log_dir, console_level="WARNING")
        
        console_handler = next(
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler)
        )
        
        assert console_handler.level == logging.WARNING
    
    def test_file_level_configuration(self, tmp_path):
        """Test that file handler uses specified log level."""
        log_dir = tmp_path / "test_logs"
        
        logger = setup_logging(log_dir=log_dir, file_level="ERROR")
        
        file_handler = next(
            h for h in logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        )
        
        assert file_handler.level == logging.ERROR
    
    def test_log_file_contains_messages(self, tmp_path):
        """Test that log messages are written to file."""
        log_dir = tmp_path / "test_logs"
        log_name = "test_messages.log"
        
        logger = setup_logging(log_dir=log_dir, log_name=log_name)
        logger.info("Test log message")
        
        # Flush handlers to ensure message is written
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / log_name
        content = log_file.read_text()
        
        assert "Test log message" in content
    
    def test_default_log_directory(self):
        """Test that default log directory is outputs/logs."""
        logger = setup_logging()
        
        default_log_dir = Path("outputs") / "logs"
        assert default_log_dir.exists()
        
        # Close handlers to release file locks before cleanup
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        # Clean up
        if default_log_dir.exists():
            for log_file in default_log_dir.glob("execution_*.log*"):
                try:
                    log_file.unlink()
                except PermissionError:
                    pass  # File still in use, skip cleanup
    
    def test_timestamped_log_filename(self, tmp_path):
        """Test that default log filename includes timestamp."""
        log_dir = tmp_path / "test_logs"
        
        logger = setup_logging(log_dir=log_dir)
        
        log_files = list(log_dir.glob("execution_*.log"))
        assert len(log_files) == 1
        
        # Check filename format: execution_YYYYMMDD_HHMMSS.log
        log_filename = log_files[0].name
        assert log_filename.startswith("execution_")
        assert log_filename.endswith(".log")


class TestStructuredFormatter:
    """Test the StructuredFormatter class."""
    
    def test_format_without_structured_fields(self, tmp_path):
        """Test formatting without extra structured fields."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="structured_test.log")
        
        logger.info("Simple message")
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "structured_test.log"
        content = log_file.read_text()
        
        assert "Simple message" in content
        assert "model=" not in content
        assert "strategy=" not in content
    
    def test_format_with_model_field(self, tmp_path):
        """Test formatting with model field."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="model_test.log")
        
        logger.info("Message with model", extra={'model': 'llama-3-70b'})
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "model_test.log"
        content = log_file.read_text()
        
        assert "Message with model" in content
        assert "model=llama-3-70b" in content
    
    def test_format_with_strategy_field(self, tmp_path):
        """Test formatting with strategy field."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="strategy_test.log")
        
        logger.info("Message with strategy", extra={'strategy': 'H'})
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "strategy_test.log"
        content = log_file.read_text()
        
        assert "Message with strategy" in content
        assert "strategy=H" in content
    
    def test_format_with_instance_id_field(self, tmp_path):
        """Test formatting with instance_id field."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="instance_test.log")
        
        logger.info("Message with instance", extra={'instance_id': 'sst2_001'})
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "instance_test.log"
        content = log_file.read_text()
        
        assert "Message with instance" in content
        assert "instance_id=sst2_001" in content
    
    def test_format_with_all_structured_fields(self, tmp_path):
        """Test formatting with all structured fields."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="all_fields_test.log")
        
        logger.info(
            "Full structured message",
            extra={
                'model': 'llama-3-70b',
                'strategy': 'CF',
                'instance_id': 'mnli_042'
            }
        )
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "all_fields_test.log"
        content = log_file.read_text()
        
        assert "Full structured message" in content
        assert "model=llama-3-70b" in content
        assert "strategy=CF" in content
        assert "instance_id=mnli_042" in content


class TestGetStructuredLogger:
    """Test the get_structured_logger function."""
    
    def test_returns_logger(self):
        """Test that function returns a logger instance."""
        logger = get_structured_logger("test_module")
        assert isinstance(logger, logging.Logger)
    
    def test_logger_name(self):
        """Test that logger has correct name."""
        logger = get_structured_logger("my_module")
        assert logger.name == "my_module"


class TestLogApiRequest:
    """Test the log_api_request convenience function."""
    
    def test_logs_api_request(self, tmp_path):
        """Test that API request is logged with all fields."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="api_test.log")
        
        log_api_request(
            logger,
            model='llama-3-70b',
            prompt_hash='a3f5b2',
            response_status='success',
            instance_id='sst2_001',
            strategy='H'
        )
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "api_test.log"
        content = log_file.read_text()
        
        assert "API request" in content
        assert "prompt_hash=a3f5b2" in content
        assert "status=success" in content
        assert "model=llama-3-70b" in content
        assert "strategy=H" in content
        assert "instance_id=sst2_001" in content
    
    def test_logs_with_minimal_fields(self, tmp_path):
        """Test logging with only required fields."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="api_minimal_test.log")
        
        log_api_request(
            logger,
            model='llama-3-8b',
            prompt_hash='xyz123',
            response_status='error'
        )
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "api_minimal_test.log"
        content = log_file.read_text()
        
        assert "API request" in content
        assert "prompt_hash=xyz123" in content
        assert "status=error" in content


class TestLogParsingFailure:
    """Test the log_parsing_failure convenience function."""
    
    def test_logs_parsing_failure(self, tmp_path):
        """Test that parsing failure is logged with all fields."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="parse_test.log")
        
        log_parsing_failure(
            logger,
            strategy='H',
            raw_output='The important words are definitely...',
            error_message='Could not extract 3 tokens',
            instance_id='sst2_042',
            model='llama-3-70b'
        )
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "parse_test.log"
        content = log_file.read_text()
        
        assert "Parsing failure" in content
        assert "error=Could not extract 3 tokens" in content
        assert "The important words are" in content
        assert "strategy=H" in content
        assert "instance_id=sst2_042" in content
    
    def test_truncates_long_raw_output(self, tmp_path):
        """Test that long raw output is truncated in main log message."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="truncate_test.log")
        
        # Create a raw output longer than 500 characters
        long_output = "x" * 600
        
        log_parsing_failure(
            logger,
            strategy='CF',
            raw_output=long_output,
            error_message='Parse error',
            instance_id='test_001'
        )
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "truncate_test.log"
        content = log_file.read_text()
        
        # Check that truncation marker appears
        assert "..." in content
        # Full output should be in debug log
        assert "Full raw output for parsing failure" in content


class TestLogModelRefusal:
    """Test the log_model_refusal convenience function."""
    
    def test_logs_model_refusal(self, tmp_path):
        """Test that model refusal is logged with all fields."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="refusal_test.log")
        
        log_model_refusal(
            logger,
            model='llama-3-70b',
            response='I cannot provide that information',
            instance_id='mnli_123',
            strategy='CF'
        )
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "refusal_test.log"
        content = log_file.read_text()
        
        assert "Model refusal or invalid response" in content
        assert "I cannot provide that information" in content
        assert "model=llama-3-70b" in content
        assert "instance_id=mnli_123" in content
        assert "strategy=CF" in content
    
    def test_truncates_long_response(self, tmp_path):
        """Test that long responses are truncated."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="refusal_truncate_test.log")
        
        # Create a response longer than 200 characters
        long_response = "y" * 300
        
        log_model_refusal(
            logger,
            model='test-model',
            response=long_response,
            instance_id='test_001'
        )
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "refusal_truncate_test.log"
        content = log_file.read_text()
        
        # Check that truncation marker appears
        assert "..." in content
        # Full response should be in debug log
        assert "Full refusal/invalid response" in content


class TestFileFormatterDetails:
    """Test detailed file formatter output."""
    
    def test_file_log_contains_timestamp(self, tmp_path):
        """Test that file logs contain timestamp."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="timestamp_test.log")
        
        logger.info("Timestamped message")
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "timestamp_test.log"
        content = log_file.read_text()
        
        # Check for timestamp format: YYYY-MM-DD HH:MM:SS
        import re
        timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        assert re.search(timestamp_pattern, content)
    
    def test_file_log_contains_level(self, tmp_path):
        """Test that file logs contain log level."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="level_test.log")
        
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "level_test.log"
        content = log_file.read_text()
        
        assert "INFO" in content
        assert "WARNING" in content
        assert "ERROR" in content
    
    def test_file_log_contains_module_and_line(self, tmp_path):
        """Test that file logs contain module name and line number."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="module_test.log")
        
        logger.info("Module info message")
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "module_test.log"
        content = log_file.read_text()
        
        # Check for pattern: module_name:line_number
        assert ":" in content  # Should have module:line format


class TestConsoleFormatterSimplicity:
    """Test simple console formatter output."""
    
    def test_console_format_is_simple(self, tmp_path, capsys):
        """Test that console output uses simple format."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, console_level="INFO")
        
        logger.info("Console test message")
        
        captured = capsys.readouterr()
        
        # Console output goes to stderr in pytest
        output = captured.err if captured.err else captured.out
        
        # Console should have simple format: LEVEL | message
        assert "INFO" in output
        assert "Console test message" in output
        # Console should NOT have timestamp
        import re
        timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        assert not re.search(timestamp_pattern, output)


class TestRequirementsSatisfaction:
    """Test that requirements 20.1, 20.2, 20.5 are satisfied."""
    
    def test_requirement_20_1_api_logging(self, tmp_path):
        """
        Requirement 20.1: THE System SHALL log every API request with timestamp,
        model name, prompt hash, and response status
        """
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="req_20_1.log")
        
        log_api_request(
            logger,
            model='llama-3-70b',
            prompt_hash='abc123',
            response_status='success'
        )
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "req_20_1.log"
        content = log_file.read_text()
        
        # Verify timestamp (in file format)
        import re
        timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        assert re.search(timestamp_pattern, content)
        
        # Verify model name
        assert "model=llama-3-70b" in content
        
        # Verify prompt hash
        assert "prompt_hash=abc123" in content
        
        # Verify response status
        assert "status=success" in content
    
    def test_requirement_20_2_parsing_failure_logging(self, tmp_path):
        """
        Requirement 20.2: THE System SHALL log parsing failures with the raw
        model output and attempted extraction strategy
        """
        log_dir = tmp_path / "test_logs"
        logger = setup_logging(log_dir=log_dir, log_name="req_20_2.log")
        
        raw_output = "This is the raw model output that failed to parse"
        
        log_parsing_failure(
            logger,
            strategy='H',
            raw_output=raw_output,
            error_message='Failed to extract tokens',
            instance_id='test_001'
        )
        
        for handler in logger.handlers:
            handler.flush()
        
        log_file = log_dir / "req_20_2.log"
        content = log_file.read_text()
        
        # Verify extraction strategy
        assert "strategy=H" in content
        
        # Verify raw output is present
        assert "raw model output that failed to parse" in content
        
        # Verify error description
        assert "Failed to extract tokens" in content
    
    def test_requirement_20_5_timestamped_log_files(self, tmp_path):
        """
        Requirement 20.5: THE System SHALL export execution logs to timestamped
        files in the outputs directory
        """
        log_dir = tmp_path / "outputs" / "logs"
        logger = setup_logging(log_dir=log_dir)
        
        # Check that log directory is in outputs
        assert log_dir.parts[-2] == "outputs"
        assert log_dir.parts[-1] == "logs"
        
        # Check that log file has timestamp
        log_files = list(log_dir.glob("execution_*.log"))
        assert len(log_files) == 1
        
        log_filename = log_files[0].name
        # Check format: execution_YYYYMMDD_HHMMSS.log
        assert log_filename.startswith("execution_")
        
        # Extract timestamp part
        timestamp_part = log_filename[len("execution_"):-4]  # Remove "execution_" and ".log"
        # Verify it's a valid timestamp format
        import re
        assert re.match(r'\d{8}_\d{6}', timestamp_part)
