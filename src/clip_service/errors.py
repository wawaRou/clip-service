class ServiceError(Exception):
    """An expected service failure that can be reported to an API caller."""

    def __init__(self, message: str, status: int = 400, code: str = "bad_request") -> None:
        super().__init__(message)
        self.status = status
        self.code = code
