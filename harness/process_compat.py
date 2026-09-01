"""为课程中的 Bash 子进程提供跨平台启动与进程树回收。"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _git_bash() -> Path | None:
    """在 Windows 上优先寻找 Git for Windows 自带的 Bash。"""
    if os.name != "nt":
        return None

    candidates: list[Path] = []
    git = shutil.which("git")
    if git:
        git_root = Path(git).resolve().parent.parent
        candidates.append(git_root / "bin" / "bash.exe")

    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(variable)
        if root:
            candidates.append(Path(root) / "Git" / "bin" / "bash.exe")

    return next((path for path in candidates if path.is_file()), None)


def shell_invocation(command: str) -> tuple[str | list[str], bool]:
    """返回适合当前平台的 ``Popen`` 命令和 ``shell`` 参数。"""
    bash = _git_bash()
    if bash is not None:
        return [str(bash), "-lc", command], False
    return command, True


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL


def attach_kill_on_close_job(process: subprocess.Popen[Any]) -> bool:
    """把 Windows 子进程放入随宿主关闭而终止的作业对象。"""
    if os.name != "nt":
        return False

    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        return False

    information = _ExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    configured = _kernel32.SetInformationJobObject(
        job,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(information),
        ctypes.sizeof(information),
    )
    assigned = configured and _kernel32.AssignProcessToJobObject(
        job, wintypes.HANDLE(process._handle)
    )
    if not assigned:
        _kernel32.CloseHandle(job)
        return False

    process._kill_on_close_job = job
    return True


def terminate_process_tree(process: subprocess.Popen[Any]) -> bool:
    """终止已绑定的 Windows 作业对象；成功接管时返回 ``True``。"""
    if os.name != "nt":
        return False
    job = getattr(process, "_kill_on_close_job", None)
    if not job:
        return False
    _kernel32.TerminateJobObject(job, 1)
    return True


def close_process_job(process: subprocess.Popen[Any]) -> None:
    """关闭作业对象句柄，并清理仍存活的后代进程。"""
    if os.name != "nt":
        return
    job = getattr(process, "_kill_on_close_job", None)
    if not job:
        return
    _kernel32.CloseHandle(job)
    process._kill_on_close_job = None
