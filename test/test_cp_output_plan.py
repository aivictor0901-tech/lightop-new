# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of shape-only plans and graph metadata ownership."""

import ast
from collections import OrderedDict
from contextlib import contextmanager
import gc
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import weakref

import pytest


class FakeTensor:
    is_cuda = True
    requires_grad = False
    ndim = 2
    device = SimpleNamespace(index=0)

    def __init__(self, rows):
        self.shape = (rows, 19)

    def is_contiguous(self):
        return True

    def numel(self):
        return self.shape[0] * 19


class FakeIndex:
    def record_stream(self, stream):
        pass


class FakeEvent:
    def record(self, stream):
        pass


class FakeCuda:
    capturing = False
    Event = FakeEvent

    def current_stream(self, device):
        return SimpleNamespace(wait_event=lambda event: None)

    def is_current_stream_capturing(self):
        return self.capturing

    @contextmanager
    def device(self, device):
        yield


def load_api():
    source = Path(__file__).resolve().parents[1] / "python/lightop/attention/cp_output.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))]
    cuda = FakeCuda()
    calls = []
    torch = SimpleNamespace(Tensor=FakeTensor, int64="int64", cuda=cuda,
                            tensor=lambda *args, **kwargs: FakeIndex())
    native = SimpleNamespace(interleave_rows_=lambda data, cp, indices: calls.append(id(indices)))
    namespace = {"torch": torch, "_cp_output": native, "OrderedDict": OrderedDict, "Lock": Lock}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace, cuda, calls


def test_cycle_plans_match_rank_to_token_order():
    namespace, _, _ = load_api()
    for cp in range(1, 18):
        for local in range(130):
            rows = cp * local
            actual = list(range(rows))
            for leader in namespace["_cycle_leaders"](rows, cp):
                first, current = actual[leader], leader
                following = (current % cp) * local + current // cp
                while following != leader:
                    actual[current] = actual[following]
                    current = following
                    following = (current % cp) * local + current // cp
                actual[current] = first
            assert actual == [rank * local + token for token in range(local) for rank in range(cp)]


def test_captured_plan_is_reused_after_lru_eviction():
    namespace, cuda, calls = load_api()
    op = namespace["interleave_cp_output_"]
    op(FakeTensor(24), 3)
    key = (0, 24, 3)
    original = weakref.ref(namespace["_PLANS"][key][0])
    cuda.capturing = True
    op(FakeTensor(24), 3)
    original_identity = calls[-1]
    cuda.capturing = False
    for local in range(33, 80):
        op(FakeTensor(local * 3), 3)
    assert key not in namespace["_PLANS"]
    cuda.capturing = True
    op(FakeTensor(24), 3)
    gc.collect()
    assert original() is not None
    assert calls[-1] == original_identity


def test_cold_capture_requires_warmup():
    namespace, cuda, calls = load_api()
    cuda.capturing = True
    with pytest.raises(RuntimeError, match="warm the CP output shape"):
        namespace["interleave_cp_output_"](FakeTensor(24), 3)
    assert not calls
