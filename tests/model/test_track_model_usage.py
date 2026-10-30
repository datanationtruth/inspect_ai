from inspect_ai._util._async import tg_collect
from inspect_ai.model import (
    Model,
    ModelOutput,
    ModelUsage,
    get_model,
    track_model_usage,
)
from inspect_ai.model._model import init_model_usage, model_usage


def _model(name: str, input_tokens: int, output_tokens: int, calls: int) -> Model:
    output = ModelOutput.from_content(name, "ok")
    output.usage = ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    return get_model(
        f"mockllm/{name}",
        custom_outputs=[output.model_copy(deep=True) for _ in range(calls)],
        memoize=False,
    )


async def test_track_model_usage_by_model() -> None:
    first = _model("first", 10, 2, calls=2)
    second = _model("second", 5, 1, calls=1)

    with track_model_usage() as usage:
        await first.generate("hi")
        await first.generate("hi")
        await second.generate("hi")

    assert usage["mockllm/first"].input_tokens == 20
    assert usage["mockllm/first"].output_tokens == 4
    assert usage["mockllm/second"].total_tokens == 6


async def test_track_model_usage_excludes_calls_outside_block() -> None:
    model = _model("model", 10, 2, calls=2)

    await model.generate("before")
    with track_model_usage() as usage:
        await model.generate("inside")

    assert usage["mockllm/model"].total_tokens == 12


async def test_track_model_usage_nested() -> None:
    model = _model("model", 10, 2, calls=2)

    with track_model_usage() as outer:
        await model.generate("outer")
        with track_model_usage() as inner:
            await model.generate("inner")

    assert inner["mockllm/model"].total_tokens == 12
    assert outer["mockllm/model"].total_tokens == 24


async def test_track_model_usage_includes_started_tasks() -> None:
    model = _model("model", 10, 2, calls=3)

    async def call() -> ModelOutput:
        return await model.generate("hi")

    with track_model_usage() as usage:
        await tg_collect([call, call, call])

    assert usage["mockllm/model"].total_tokens == 36


async def test_track_model_usage_keeps_eval_usage() -> None:
    model = _model("model", 10, 2, calls=1)
    init_model_usage({})

    with track_model_usage():
        await model.generate("hi")

    assert model_usage()["mockllm/model"].total_tokens == 12
