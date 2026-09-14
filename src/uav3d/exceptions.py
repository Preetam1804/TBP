"""Exception hierarchy for uav3d."""


class Uav3dError(Exception):
    """Base exception for all uav3d errors."""



class ConfigError(Uav3dError):
    """Raised when configuration loading, parsing, or validation fails."""



class DataError(Uav3dError):
    """Raised when dataset input files or directory structures are missing or malformed."""



class ValidationError(Uav3dError):
    """Raised when schema validation or trajectory validation fails."""



class PipelineError(Uav3dError):
    """Raised when an internal pipeline stage encounters an unrecoverable failure."""

