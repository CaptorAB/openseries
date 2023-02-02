class OpenTimeSeriesValidationError(Exception):
    """Top-level error for validating.
    This exception should normally not be raised, only subclasses of this
    exception."""

    def __str__(self) -> str:
        """Return the exception message."""
        return "".join(self.args[:1]) or getattr(self, "message", "")


class FromFixedRateDatesInputError(OpenTimeSeriesValidationError):
    """Something is wrong with the arguments."""

    message = "If d_range is not provided both days and end_dt must be."
