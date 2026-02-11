from __future__ import annotations

import threading
from unittest.mock import patch

from transformers.generation import GenerationMixin
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
)

from chroma.engine.streaming_engine import AbortOnCancelCriteria
from chroma.generation_chroma import ChromaGenerationMixin


class _DummyGenerationMixin(ChromaGenerationMixin):
    pass


def test_keeps_custom_cancel_stopping_criteria() -> None:
    dummy = _DummyGenerationMixin()
    cancel = AbortOnCancelCriteria(threading.Event())
    max_length = MaxLengthCriteria(max_length=128)
    criteria = StoppingCriteriaList([max_length, cancel])

    with patch.object(GenerationMixin, "_get_stopping_criteria", return_value=criteria):
        kept = dummy._get_stopping_criteria()

    assert len(kept) == 2
    assert kept[0] is max_length
    assert cancel in kept


def test_filters_unsupported_transformers_stopping_criteria() -> None:
    dummy = _DummyGenerationMixin()
    max_length = MaxLengthCriteria(max_length=64)
    max_time = MaxTimeCriteria(max_time=1.0, initial_timestamp=0.0)
    criteria = StoppingCriteriaList([max_length, max_time])

    with patch.object(GenerationMixin, "_get_stopping_criteria", return_value=criteria):
        kept = dummy._get_stopping_criteria()

    assert len(kept) == 1
    assert kept[0] is max_length
