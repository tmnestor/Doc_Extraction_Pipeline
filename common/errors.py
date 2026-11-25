"""The error every setting that cannot be used is refused with."""


class ConfigError(ValueError):
    """The settings cannot be used. The message says what, where, what is expected and the fix."""
