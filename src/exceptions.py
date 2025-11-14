
class DataError(Exception):
    """Raised when required columns are missing or data can't be processed."""

class ModelError(Exception):
    """Raised when model training or inference fails."""
