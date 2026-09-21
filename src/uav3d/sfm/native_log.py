"""Capture of the native (C++) stderr stream emitted by the pycolmap binary.

pycolmap 3.12.x logs its internal glog output (e.g. 'Processed file [i/N]',
'Discarding reconstruction due to insufficient size') directly to file
descriptor 2. There is no Python API to change that level, so we temporarily
redirect fd 2 to a log file for the duration of a COLMAP call. Python-level
print/log output is unaffected because it goes through a separate file
object that is captured before the redirect.

The redirect is best-effort: if the platform refuses the fd swap (should
never happen on POSIX or Windows CRT), the call proceeds with console logging.
"""

import os
import sys
from pathlib import Path
from typing import Self


class NativeStderrCapture:
    """Context manager that redirects native fd 2 to a file for its duration."""

    def __init__(self, log_path: str | Path | None) -> None:
        self.log_path = Path(log_path) if log_path is not None else None
        self._file = None
        self._saved_fd = None
        self._py_stderr = sys.stderr
        self._active = False

    def __enter__(self) -> Self:
        if self.log_path is None:
            return self
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.log_path, "ab", buffering=0)
            self._saved_fd = os.dup(2)
            os.dup2(self._file.fileno(), 2)
            self._active = True
        except OSError:
            self._restore()
            print(
                "[uav3d] Native stderr capture unavailable on this platform; "
                "COLMAP log lines will remain on the console.",
                file=self._py_stderr,
            )
        return self

    def _restore(self) -> None:
        if self._active:
            try:
                os.dup2(self._saved_fd, 2)
            except OSError:
                pass
            try:
                os.close(self._saved_fd)
            except OSError:
                pass
            self._saved_fd = None
            self._active = False
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

    def __exit__(self, exc_type, exc, tb) -> None:
        self._restore()
