#!/usr/bin/env python3
"""SGLang Engine wrapper that installs the WP3A direct-eviction probe."""
from __future__ import annotations

import os
import sys

from sglang.srt.entrypoints.engine import Engine as _SglangEngine
from sglang.srt.managers import scheduler as _scheduler_module


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from targeted_probe import install_control_server


CONTROL_PORT = int(os.environ.get("WP3D_CTRL_PORT", "49935"))
_original_run_scheduler_process = _scheduler_module.run_scheduler_process


def _wrapped_run_scheduler_process(*args, **kwargs):
    """Install the queue and idle-thread drain inside the scheduler process."""
    scheduler_cls = _scheduler_module.Scheduler

    original_init = scheduler_cls.__init__
    if not getattr(original_init, "_wp3d_patched", False):

        def patched_init(self, *init_args, **init_kwargs):
            original_init(self, *init_args, **init_kwargs)
            port = install_control_server(self, CONTROL_PORT)
            print(
                f"[FSWP3D] probe_up port={port} "
                f"tree_cache={type(self.tree_cache).__name__}",
                flush=True,
            )

        patched_init._wp3d_patched = True
        scheduler_cls.__init__ = patched_init

    original_on_idle = scheduler_cls.on_idle
    if not getattr(original_on_idle, "_wp3d_patched", False):

        def patched_on_idle(self):
            state = getattr(self, "_wp3d_probe_state", None)
            if state is not None and self.is_fully_idle():
                state.drain_one(self)
            # The ordinary on_idle path immediately runs SGLang's pool and
            # radix-tree invariant checks after a control mutation.
            return original_on_idle(self)

        patched_on_idle._wp3d_patched = True
        scheduler_cls.on_idle = patched_on_idle

    return _original_run_scheduler_process(*args, **kwargs)


class DirectEvictionEngine(_SglangEngine):
    run_scheduler_process_func = staticmethod(_wrapped_run_scheduler_process)

