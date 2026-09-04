# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitOverride=false, reportUnusedCallResult=false
# pyright: reportPrivateUsage=false, reportIncompatibleVariableOverride=false

from asyncio import Task
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
)
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Generic, TypeVar
import asyncio

__all__ = (
    "Hangup",
    "Channel",
    "asyncify",
    "amerge",
    "as_completed",
)


T = TypeVar("T")
U = TypeVar("U")


class Hangup(Exception):
    pass


@dataclass(slots=True)
class Channel(Generic[T]):
    @dataclass(slots=True)
    class Refcount:
        _count: int = 0
        _gone: asyncio.Event = field(default_factory=asyncio.Event)

        def incr(self) -> None:
            self._count += 1
            self._gone.clear()

        def decr(self) -> None:
            self._count -= 1
            if not self._count:
                self._gone.set()

    _queue: asyncio.Queue[T] = field(default_factory=asyncio.Queue)
    _readers: Refcount = field(default_factory=Refcount)
    _writers: Refcount = field(default_factory=Refcount)

    async def _race(self, awaitable: Awaitable[U], against: Refcount) -> U:
        task = asyncio.ensure_future(awaitable)
        waiter = asyncio.ensure_future(against._gone.wait())
        try:
            await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
            if task.done():
                return task.result()
            raise Hangup
        finally:
            for future in (task, waiter):
                future.cancel()
                with suppress(BaseException):
                    await future

    @asynccontextmanager
    async def reader(self) -> AsyncGenerator[AsyncGenerator[T, None], None]:
        async def reads() -> AsyncGenerator[T, None]:
            with suppress(Hangup):
                while True:
                    yield await self._race(self._queue.get(), self._writers)

        self._readers.incr()
        try:
            yield reads()
        finally:
            self._readers.decr()

    @asynccontextmanager
    async def writer(self) -> AsyncGenerator[Callable[[T], Awaitable[None]], None]:
        async def put(item: T) -> None:
            if self._readers._gone.is_set():
                raise Hangup
            await self._race(self._queue.put(item), self._readers)

        self._writers.incr()
        try:
            yield put
        finally:
            self._writers.decr()


def asyncify(iterable: Iterable[T] | AsyncIterable[T]) -> AsyncIterator[T]:
    if isinstance(iterable, AsyncIterable):
        return aiter(iterable)

    async def agen() -> AsyncGenerator[T, None]:
        for item in iterable:
            yield item

    return agen()


async def amerge(*iterables: Iterable[T] | AsyncIterable[T]) -> AsyncGenerator[T, None]:
    pending = {
        source: asyncio.ensure_future(anext(source))
        for source in map(asyncify, iterables)
    }

    try:
        while pending:
            await asyncio.wait(pending.values(), return_when=asyncio.FIRST_COMPLETED)

            source = next(source for source, task in pending.items() if task.done())
            try:
                yield pending[source].result()
            except StopAsyncIteration:
                del pending[source]
            else:
                pending[source] = asyncio.ensure_future(anext(source))
    finally:
        for task in pending.values():
            task.cancel()
        for task in pending.values():
            with suppress(BaseException):
                await task
        for source in pending:
            if isinstance(source, AsyncGenerator):
                with suppress(BaseException):
                    await source.aclose()


async def as_completed(
    awaitables: Iterable[Awaitable[T]] | AsyncIterable[Awaitable[T]],
    limit: int,
    delay: float = 0.0,
) -> AsyncGenerator[Task[T], None]:
    source = asyncify(awaitables)
    pending: set[Task[T]] = set()

    try:
        for _ in range(limit):
            try:
                awaitable = await anext(source)
                pending.add(asyncio.ensure_future(awaitable))
            except StopAsyncIteration:
                break

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                yield task

                await asyncio.sleep(delay)

                try:
                    awaitable = await anext(source)
                    pending.add(asyncio.ensure_future(awaitable))
                except StopAsyncIteration:
                    pass
    except BaseException:
        for task in pending:
            task.cancel()
        for task in pending:
            with suppress(BaseException):
                await task
        raise
    finally:
        if isinstance(source, AsyncGenerator):
            await source.aclose()
