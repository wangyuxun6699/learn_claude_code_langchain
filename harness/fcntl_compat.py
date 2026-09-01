"""课程所需的最小 ``fcntl.flock`` 跨平台兼容层。

Unix 直接代理 ``fcntl``；Windows 使用 ``msvcrt.locking`` 锁定锁文件的
第一个字节。只实现课程实际使用的 EX / NB / UN 三种标志。
"""
from __future__ import annotations

import os

if os.name != "nt":
    from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock
else:
    import msvcrt

    LOCK_EX = 1
    LOCK_NB = 4
    LOCK_UN = 8

    def flock(file_descriptor: int, operation: int) -> None:
        """用 Windows 字节区间锁模拟课程使用的排他 flock。"""
        position = os.lseek(file_descriptor, 0, os.SEEK_CUR)
        try:
            if os.fstat(file_descriptor).st_size == 0:
                os.lseek(file_descriptor, 0, os.SEEK_SET)
                os.write(file_descriptor, b"\0")
            os.lseek(file_descriptor, 0, os.SEEK_SET)
            if operation & LOCK_UN:
                mode = msvcrt.LK_UNLCK
            elif operation & LOCK_NB:
                mode = msvcrt.LK_NBLCK
            else:
                mode = msvcrt.LK_LOCK
            try:
                msvcrt.locking(file_descriptor, mode, 1)
            except OSError as exc:
                if operation & LOCK_UN:
                    return
                if operation & LOCK_NB:
                    raise BlockingIOError(str(exc)) from exc
                raise
        finally:
            os.lseek(file_descriptor, position, os.SEEK_SET)


__all__ = ["LOCK_EX", "LOCK_NB", "LOCK_UN", "flock"]
