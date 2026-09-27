# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitOverride=false, reportUnusedCallResult=false
# pyright: reportPrivateUsage=false, reportIncompatibleVariableOverride=false

from asyncio import Task
from collections.abc import AsyncGenerator, Awaitable
from typing import Any, TypeVar
import asyncio

T = TypeVar("T")


async def cancel_all(*futures: "asyncio.Future[Any]") -> None:
    for future in futures:
        future.cancel()

    for future in futures:
        if not future.done():
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()

            def release(
                _: "asyncio.Future[Any]", waiter: asyncio.Future[None] = waiter
            ) -> None:
                if not waiter.done():
                    waiter.set_result(None)

            future.add_done_callback(release)
            try:
                await waiter
            finally:
                future.remove_done_callback(release)

        if not future.cancelled():
            future.exception()


async def as_completed(
    source: AsyncGenerator[Awaitable[T], None],
    limit: int,
    delay: float = 0.0,
) -> AsyncGenerator[Task[T], None]:
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
        await cancel_all(*pending)
        raise
    finally:
        await source.aclose()
