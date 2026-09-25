"""
Exit a training process without the Windows CUDA shutdown crash.

On Windows, a process that has run a multi-layer `nn.LSTM` with dropout in
training mode on the GPU crashes as it exits, with status 0xC0000409, after
all of its work is done. cuDNN keeps the RNN dropout state in a static cache,
and destroying that cache while the process unloads its DLLs fails. Both
branches of every model here carry such an LSTM, so every training run hit it,
and `run.py` recorded finished runs as failed.

Neither `sys.exit` nor `os._exit` avoids it, because both still unload the
DLLs. Ending the process with `TerminateProcess` does: the exit code is the one
given, and the DLL teardown never runs. Everything the run writes is closed
before that point, and the output streams are flushed first.

Elsewhere, or when CUDA was never used, this is an ordinary exit.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Callable


def exit_without_teardown(code: int = 0) -> None:
    """Flush the output streams and end the process with `code`."""
    sys.stdout.flush()
    sys.stderr.flush()
    torch = sys.modules.get("torch")
    if os.name == "nt" and torch is not None and torch.cuda.is_initialized():
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), code)
    # Reached only when termination was not needed or did not happen.
    sys.exit(code)


def run_main(main: Callable[[], None]) -> None:
    """Run `main`, then exit with its status and without the teardown.

    A normal return exits 0. An exception prints its traceback and exits 1,
    and `SystemExit` keeps its own status, so a failure is still reported as
    one.
    """
    try:
        main()
    except SystemExit as stop:
        if stop.code is None:
            code = 0
        elif isinstance(stop.code, int):
            code = stop.code
        else:
            print(stop.code, file=sys.stderr)
            code = 1
    except BaseException:
        traceback.print_exc()
        code = 1
    else:
        code = 0
    exit_without_teardown(code)
