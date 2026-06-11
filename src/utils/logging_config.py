import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    def format(self, record):
        extra_fields = {}
        for key in ('model', 'strategy', 'instance_id', 'status', 'prompt_hash', 'response_status', 'error', 'raw_output', 'dataset'):
            val = getattr(record, key, None)
            if val is not None:
                extra_fields[key] = val
        log_msg = super().format(record)
        if extra_fields:
            parts = ' | '.join(f'{k}={v}' for k, v in extra_fields.items())
            log_msg = f'{log_msg} | {parts}'
        return log_msg


def setup_logging(log_dir: str = "outputs/logs", log_name: str = None,
                  console_level: str = "INFO", file_level: str = "DEBUG",
                  max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    if log_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"execution_{timestamp}.log"
    log_file = log_path / log_name

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = RotatingFileHandler(str(log_file), maxBytes=max_bytes, backupCount=backup_count)
    file_format = StructuredFormatter('%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s')
    file_handler.setFormatter(file_format)
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_format = StructuredFormatter('%(levelname)-8s | %(message)s')
    console_handler.setFormatter(console_format)
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    logger.addHandler(console_handler)

    return logger


def get_structured_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_api_request(logger, *, model, prompt_hash, response_status, instance_id=None, strategy=None, **kwargs):
    extra = {'model': model, 'prompt_hash': prompt_hash, 'response_status': response_status}
    if instance_id:
        extra['instance_id'] = instance_id
    if strategy:
        extra['strategy'] = strategy
    extra.update(kwargs)
    logger.info("API request", extra=extra)


def log_parsing_failure(logger, *, strategy, raw_output, error_message, instance_id=None, model=None):
    extra = {'strategy': strategy, 'error': error_message}
    if instance_id:
        extra['instance_id'] = instance_id
    if model:
        extra['model'] = model
    truncated = raw_output[:500] + "..." if len(raw_output) > 500 else raw_output
    logger.warning(f"Parsing failure | raw={truncated}", extra=extra)
    logger.debug(f"Full raw output for parsing failure: {raw_output}", extra=extra)


def log_model_refusal(logger, *, model, response, instance_id=None, strategy=None):
    extra = {'model': model}
    if instance_id:
        extra['instance_id'] = instance_id
    if strategy:
        extra['strategy'] = strategy
    truncated = response[:200] + "..." if len(response) > 200 else response
    logger.warning(f"Model refusal or invalid response: {truncated}", extra=extra)
    logger.debug(f"Full refusal/invalid response: {response}", extra=extra)
