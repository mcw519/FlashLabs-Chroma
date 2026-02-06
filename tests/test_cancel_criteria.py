from __future__ import annotations

import threading

import torch

from chroma.engine.streaming_engine import AbortOnCancelCriteria


def test_abort_on_cancel_criteria() -> None:
    cancel_event = threading.Event()
    criteria = AbortOnCancelCriteria(cancel_event)

    should_continue = criteria(torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1)))
    assert should_continue is False

    cancel_event.set()
    should_stop = criteria(torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1)))
    assert should_stop is True
