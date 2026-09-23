"""A stand-in `wandb` module for the tests: records every call and never reaches the network.

Inject it with `monkeypatch.setitem(sys.modules, "wandb", stub_wandb())`; the code under test then
imports it as it would import the real package.

`fail` makes one call raise; `gate` makes calls wait on a `threading.Event` (a W&B that hangs until
the test lets it go); `delay` makes every call of a method sleep. `mod.released` records whether the
logger unregistered the SDK's exit-time teardown, which is reached as
`wandb.sdk.wandb_setup._singleton._connection._cleanup` in SDK 0.30.

`exit_hook=True` also registers that teardown the way SDK 0.30 does, at the start of `init` and before
anything can fail: at interpreter exit it waits (for `HANG` seconds) unless the run was finished or
the teardown was unregistered. `exit_hook="late"` registers it, and only then creates the connection
that unregisters it, once a gated `init` is let go: a teardown that appears after the logger has
already given up. Use either only in a subprocess, never in the test process itself.
"""
import atexit
import threading
import time
import types

HANG = 60.0     # the longest a gated call waits before giving up, so a broken test cannot hang the suite


class FakeRun:
    """Records what a W&B run was asked to do; `fail` names the method that raises."""

    def __init__(self, calls, fail=None, gate=None, delay=None, on_finish=None):
        self.calls, self.fail, self.gate, self.delay = calls, fail, gate or {}, delay or {}
        self.on_finish = on_finish

    def _wait(self, name):
        if name in self.gate:
            self.gate[name].wait(HANG)
        if name in self.delay:
            time.sleep(self.delay[name])

    def define_metric(self, *a, **k):
        self.calls.append(("define_metric", a, k))

    def log(self, row, **k):
        self._wait("log")
        if self.fail == "log":
            raise RuntimeError("W&B is unreachable")
        self.calls.append(("log", dict(row), k))

    def finish(self, *a, **k):
        self._wait("finish")
        if self.fail == "finish":
            raise RuntimeError("W&B is unreachable")
        self.calls.append(("finish", a, k))
        if self.on_finish:
            self.on_finish()


def stub_wandb(fail=None, gate=None, delay=None, exit_hook=False):
    """A `wandb` module that records calls. `fail` in {"init", "log", "finish"} makes that call raise;
    `gate` and `delay` map the same names to an Event to wait on and to seconds to sleep; `exit_hook`
    registers the SDK-shaped exit-time teardown (see the module docstring)."""
    mod = types.ModuleType("wandb")
    mod.calls = []
    mod.released = []
    mod.finished = False
    gate, delay = gate or {}, delay or {}

    def teardown():
        if not mod.finished:
            time.sleep(HANG)        # the SDK waiting on its service for a run nobody finished

    def finished():
        mod.finished = True

    def register():
        atexit.unregister(teardown)
        atexit.register(teardown)
        mod.sdk.wandb_setup._singleton._connection = conn

    def init(**kw):
        mod.calls.append(("init", (), kw))
        if exit_hook is True:
            register()
        if "init" in gate:
            gate["init"].wait(HANG)
        if "init" in delay:
            time.sleep(delay["init"])
        if exit_hook == "late":
            register()
        if fail == "init":
            raise RuntimeError("W&B is down")
        return FakeRun(mod.calls, fail, gate, delay, on_finish=finished)

    def cleanup():
        mod.released.append(threading.current_thread().name)
        atexit.unregister(teardown)

    mod.init = init
    mod.Settings = lambda **kw: dict(kw)
    conn = types.SimpleNamespace(_cleanup=cleanup)
    mod.sdk = types.SimpleNamespace(wandb_setup=types.SimpleNamespace(
        _singleton=types.SimpleNamespace(_connection=None if exit_hook == "late" else conn)))
    return mod


def logged(mod):
    """The rows a stub run was asked to log, in order."""
    return [row for kind, row, _ in mod.calls if kind == "log"]


def inits(mod):
    """The keyword arguments of every `wandb.init` call."""
    return [kw for kind, _, kw in mod.calls if kind == "init"]
