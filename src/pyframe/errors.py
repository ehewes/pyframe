from __future__ import annotations


class PyFrameError(Exception):
    pass


class UnsupportedMediaError(PyFrameError):
    pass


class MediaDecodeError(PyFrameError):
    pass


class BackendUnavailableError(PyFrameError):
    def __init__(self, backend: str, install_hint: str):
        self.backend = backend
        self.install_hint = install_hint
        super().__init__(
            f"The '{backend}' backend is not available. Install it with: {install_hint}"
        )
