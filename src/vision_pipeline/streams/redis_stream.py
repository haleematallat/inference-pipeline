import json
from typing import Any

from vision_pipeline.common.results import InferResult
from vision_pipeline.exceptions import StreamError
from vision_pipeline.streams.base import IStreamHandler


class RedisStreamHandler(IStreamHandler):
    def __init__(
        self,
        redis_url: str,
        input_queue: str,
        output_queue: str,
        capacity: int = 64,
    ):
        super().__init__()
        self.redis_url = redis_url
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.capacity = capacity
        self.client: Any = None

    def start(self) -> None:
        try:
            from redis import Redis
            from redis.exceptions import ConnectionError as RedisConnectionError
        except ImportError as error:
            raise StreamError("Redis support requires installation with the 'redis' extra") from error
        try:
            self.client = Redis.from_url(self.redis_url)
            self.client.ping()
        except (OSError, RedisConnectionError) as error:
            raise StreamError("Could not connect to Redis") from error
        self.started = True

    def get_data(self, timeout: float = 0) -> InferResult | None:
        if not self.started or self.client is None:
            raise RuntimeError("stream handler has not been started")
        item = self.client.blpop(self.input_queue, timeout=timeout)
        if item is None:
            return None
        try:
            return InferResult.from_dict(json.loads(item[1]))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise StreamError("Invalid input payload from Redis") from error

    def push_data(self, infer_result: InferResult) -> None:
        if not self.started or self.client is None:
            raise RuntimeError("stream handler has not been started")
        payload = json.dumps(infer_result.to_dict(), sort_keys=True)
        pipeline = self.client.pipeline()
        pipeline.rpush(self.output_queue, payload)
        pipeline.ltrim(self.output_queue, -self.capacity, -1)
        pipeline.execute()

    def stop(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        self.started = False
