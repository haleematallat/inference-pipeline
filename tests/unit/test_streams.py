from queue import Full

import pytest

from vision_pipeline.common.results import InferResult
from vision_pipeline.streams.memory import InMemoryStreamHandler


def test_in_memory_stream_push_and_get():
    stream = InMemoryStreamHandler(capacity=2)
    stream.put_input(InferResult(1, "one.ppm"))
    stream.start()
    result = stream.get_data()
    assert result is not None
    assert result.source == "one.ppm"


def test_in_memory_stream_output():
    stream = InMemoryStreamHandler()
    stream.start()
    result = InferResult(1, "one.ppm")
    stream.push_data(result)
    assert stream.outputs == [result]


def test_in_memory_stream_capacity():
    stream = InMemoryStreamHandler(capacity=1)
    stream.put_input(InferResult(1, "one.ppm"))
    with pytest.raises(Full):
        stream.put_input(InferResult(2, "two.ppm"))


def test_stream_start_and_stop():
    stream = InMemoryStreamHandler()
    stream.start()
    assert stream.started is True
    stream.stop()
    assert stream.started is False


def test_empty_stream_returns_none():
    stream = InMemoryStreamHandler()
    stream.start()
    assert stream.get_data() is None
