from typing import Dict, Any, Optional


class ExplanationStudyError(Exception):
    def __init__(self, message: str, error_code: str = "ESE000", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.__str__())

    def __str__(self):
        base = f"[{self.error_code}] {self.message}"
        if self.details:
            parts = " | ".join(f"{k}={v}" for k, v in self.details.items())
            base += f" ({parts})"
        return base

    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
        }


class DataLoadError(ExplanationStudyError):
    def __init__(self, message: str, error_code: str = "DLE000", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=error_code, details=details)


class APIError(ExplanationStudyError):
    def __init__(self, message: str, error_code: str = "API000", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=error_code, details=details)


class RateLimitExhausted(APIError):
    """Raised when rate-limit retries are exhausted; triggers immediate checkpoint save."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="API429", details=details)


class DailyRateLimitExhausted(RateLimitExhausted):
    """Raised when a per-day quota (Groq RPD/TPD) is hit.

    Unlike a per-minute limit, waiting out the backoff is futile — the daily window
    won't free up for hours — so the run should stop and surface the reset time
    rather than honoring a misleading short retry-after.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        APIError.__init__(self, message, error_code="API429D", details=details)


class ParsingError(ExplanationStudyError):
    def __init__(self, message: str, error_code: str = "PRS000", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=error_code, details=details)


class ValidationError(ExplanationStudyError):
    def __init__(self, message: str, error_code: str = "VAL000", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=error_code, details=details)


class ConfigurationError(ExplanationStudyError):
    def __init__(self, message: str, error_code: str = "CFG000", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=error_code, details=details)


class PromptValidationError(ExplanationStudyError):
    def __init__(self, message: str, error_code: str = "PRV000", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=error_code, details=details)


def raise_data_load_error(message: str, error_code: str = "DLE000", **details):
    raise DataLoadError(message, error_code=error_code, details=details)


def raise_api_error(message: str, error_code: str = "API000", **details):
    raise APIError(message, error_code=error_code, details=details)


def raise_parsing_error(message: str, error_code: str = "PRS000", **details):
    raise ParsingError(message, error_code=error_code, details=details)


def raise_validation_error(message: str, error_code: str = "VAL000", **details):
    raise ValidationError(message, error_code=error_code, details=details)


def raise_configuration_error(message: str, error_code: str = "CFG000", **details):
    raise ConfigurationError(message, error_code=error_code, details=details)


def raise_prompt_validation_error(message: str, error_code: str = "PRV000", **details):
    raise PromptValidationError(message, error_code=error_code, details=details)
