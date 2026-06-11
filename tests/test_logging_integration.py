"""
Integration test demonstrating the logging configuration module in action.

This test shows how the logging module would be used in a real scenario
from the LLM Explanation Agreement Study pipeline.
"""

import pytest
from pathlib import Path
import re

from src.utils.logging_config import (
    setup_logging,
    get_structured_logger,
    log_api_request,
    log_parsing_failure,
    log_model_refusal
)


def test_realistic_inference_pipeline_logging(tmp_path):
    """
    Integration test simulating a realistic inference pipeline with logging.
    
    This demonstrates how the logging module would be used to track:
    - API requests with structured fields
    - Parsing failures with raw outputs
    - Model refusals
    """
    log_dir = tmp_path / "integration_logs"
    
    # Setup logging at the start of the pipeline
    logger = setup_logging(log_dir=log_dir, log_name="pipeline.log")
    
    # Get module-specific loggers
    inference_logger = get_structured_logger("inference_engine")
    parser_logger = get_structured_logger("parser")
    
    # Simulate processing multiple instances
    for instance_id in ["sst2_001", "sst2_002", "sst2_003"]:
        # Log classification API request
        log_api_request(
            inference_logger,
            model='llama-3-70b',
            prompt_hash=f'hash_{instance_id}',
            response_status='success',
            instance_id=instance_id
        )
        
        # Log explanation requests for different strategies
        for strategy in ['H', 'R', 'CF', 'RO']:
            log_api_request(
                inference_logger,
                model='llama-3-70b',
                prompt_hash=f'hash_{instance_id}_{strategy}',
                response_status='success',
                instance_id=instance_id,
                strategy=strategy
            )
    
    # Simulate a parsing failure
    log_parsing_failure(
        parser_logger,
        strategy='H',
        raw_output='The most important words are: great, wonderful, amazing, fantastic',
        error_message='Expected 3 tokens but found 4',
        instance_id='sst2_004',
        model='llama-3-70b'
    )
    
    # Simulate a model refusal
    log_model_refusal(
        inference_logger,
        model='llama-3-70b',
        response='I apologize, but I cannot provide a counterfactual for this content.',
        instance_id='mnli_001',
        strategy='CF'
    )
    
    # Flush all handlers
    for handler in logger.handlers:
        handler.flush()
    
    # Verify log file was created and contains expected content
    log_file = log_dir / "pipeline.log"
    assert log_file.exists()
    
    content = log_file.read_text()
    
    # Verify API request logging
    assert "API request" in content
    assert "llama-3-70b" in content
    assert "sst2_001" in content
    assert "strategy=H" in content
    assert "strategy=R" in content
    assert "strategy=CF" in content
    assert "strategy=RO" in content
    
    # Verify parsing failure logging
    assert "Parsing failure" in content
    assert "Expected 3 tokens but found 4" in content
    assert "great, wonderful, amazing" in content
    
    # Verify model refusal logging
    assert "Model refusal or invalid response" in content
    assert "cannot provide a counterfactual" in content
    
    # Verify structured fields appear correctly
    assert "model=llama-3-70b" in content
    assert "instance_id=sst2_001" in content
    
    # Verify timestamps are present
    timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
    assert len(re.findall(timestamp_pattern, content)) > 0


def test_multiple_instances_with_structured_logging(tmp_path):
    """
    Test that structured logging correctly tracks multiple instances
    being processed concurrently.
    """
    log_dir = tmp_path / "structured_logs"
    logger = setup_logging(log_dir=log_dir, log_name="structured.log")
    
    inference_logger = get_structured_logger("inference")
    
    # Simulate processing instances from different datasets
    datasets_instances = [
        ("sst2", ["sst2_001", "sst2_002"]),
        ("mnli", ["mnli_001", "mnli_002"]),
        ("ag_news", ["ag_001", "ag_002"])
    ]
    
    for dataset, instances in datasets_instances:
        for instance_id in instances:
            log_api_request(
                inference_logger,
                model='llama-3-70b',
                prompt_hash=f'hash_{instance_id}',
                response_status='success',
                instance_id=instance_id,
                dataset=dataset  # Extra field
            )
    
    # Flush handlers
    for handler in logger.handlers:
        handler.flush()
    
    log_file = log_dir / "structured.log"
    content = log_file.read_text()
    
    # Verify all instances were logged
    assert "sst2_001" in content
    assert "sst2_002" in content
    assert "mnli_001" in content
    assert "mnli_002" in content
    assert "ag_001" in content
    assert "ag_002" in content
    
    # Verify dataset info was captured as extra field
    assert "dataset=sst2" in content
    assert "dataset=mnli" in content
    assert "dataset=ag_news" in content


def test_log_rotation_simulation(tmp_path):
    """
    Test that the rotating file handler is properly configured
    (though we don't actually trigger rotation in this test).
    """
    log_dir = tmp_path / "rotation_logs"
    logger = setup_logging(log_dir=log_dir, log_name="rotation_test.log")
    
    # Write a reasonable number of log messages
    for i in range(100):
        logger.info(f"Log message number {i}")
    
    # Flush handlers
    for handler in logger.handlers:
        handler.flush()
    
    log_file = log_dir / "rotation_test.log"
    assert log_file.exists()
    
    # Verify file size is reasonable (not hitting 10MB limit with 100 messages)
    assert log_file.stat().st_size < 1024 * 1024  # Less than 1MB
    
    # Verify we can read the log
    content = log_file.read_text()
    assert "Log message number 0" in content
    assert "Log message number 99" in content


def test_error_tracking_workflow(tmp_path):
    """
    Integration test for error tracking workflow as per Requirement 20.
    """
    log_dir = tmp_path / "error_logs"
    logger = setup_logging(log_dir=log_dir, log_name="errors.log", console_level="WARNING")
    
    parser_logger = get_structured_logger("parser")
    
    # Simulate various error scenarios
    
    # 1. Parsing failure with incomplete highlighting response
    log_parsing_failure(
        parser_logger,
        strategy='H',
        raw_output='Most important: good, excellent',
        error_message='Incomplete response: expected 3 tokens, got 2',
        instance_id='sst2_100',
        model='llama-3-8b'
    )
    
    # 2. Parsing failure with malformed counterfactual
    log_parsing_failure(
        parser_logger,
        strategy='CF',
        raw_output='Here is a different version of the text...',
        error_message='Could not identify original vs counterfactual sections',
        instance_id='mnli_050',
        model='llama-3-70b'
    )
    
    # 3. Parsing failure with rank-ordering format issue
    log_parsing_failure(
        parser_logger,
        strategy='RO',
        raw_output='Important tokens include: first, second, third, and fourth',
        error_message='Could not extract rank positions',
        instance_id='ag_025',
        model='llama-3-8b'
    )
    
    # Flush handlers
    for handler in logger.handlers:
        handler.flush()
    
    log_file = log_dir / "errors.log"
    content = log_file.read_text()
    
    # Verify all parsing failures are logged
    assert content.count("Parsing failure") == 3
    
    # Verify strategy-specific errors
    assert "strategy=H" in content
    assert "strategy=CF" in content
    assert "strategy=RO" in content
    
    # Verify error messages
    assert "expected 3 tokens, got 2" in content
    assert "Could not identify original vs counterfactual sections" in content
    assert "Could not extract rank positions" in content
    
    # Verify instance IDs are tracked
    assert "sst2_100" in content
    assert "mnli_050" in content
    assert "ag_025" in content
