"""A stand-in `wandb` module for the tests: records every call and never reaches the network.

Inject it with `monkeypatch.setitem(sys.modules, "wandb", stub_wandb())`; the code under test then
imports it as it would import the real package.
"""
import types


class FakeRun:
    """Records what a W&B run was asked to do; `fail` names the method that raises."""

    def __init__(self, calls, fail=None):
        self.calls, self.fail = calls, fail

    def define_metric(self, *a, **k):
        self.calls.append(("define_metric", a, k))

    def log(self, row, **k):
        if self.fail == "log":
            raise RuntimeError("W&B is unreachable")
        self.calls.append(("log", dict(row), k))

    def finish(self, *a, **k):
        if self.fail == "finish":
            raise RuntimeError("W&B is unreachable")
        self.calls.append(("finish", a, k))


def stub_wandb(fail=None):
    """A `wandb` module that records calls. `fail` in {"init", "log", "finish"} makes that call raise."""
    mod = types.ModuleType("wandb")
    mod.calls = []

    def init(**kw):
        mod.calls.append(("init", (), kw))
        if fail == "init":
            raise RuntimeError("W&B is down")
        return FakeRun(mod.calls, fail)

    mod.init = init
    mod.Settings = lambda **kw: dict(kw)
    return mod


def logged(mod):
    """The rows a stub run was asked to log, in order."""
    return [row for kind, row, _ in mod.calls if kind == "log"]


def inits(mod):
    """The keyword arguments of every `wandb.init` call."""
    return [kw for kind, _, kw in mod.calls if kind == "init"]
