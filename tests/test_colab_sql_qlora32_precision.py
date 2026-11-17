"""CPU-only regression tests for T4 trainable adapter precision preflight.

Fake tensors keep GitHub CI free of torch/GPU; physical T4 behavior is unverified.
"""
from scripts.colab_sql_qwen9b_qlora32 import normalize_t4_trainable_precision


class FakeTorch:
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"


class FakeTensor:
    def __init__(self, dtype):
        self.dtype = dtype

    def to(self, dtype):
        return FakeTensor(dtype)


class FakeParam:
    def __init__(self, dtype, *, requires_grad=True):
        self.data = FakeTensor(dtype)
        self.requires_grad = requires_grad

    @property
    def dtype(self):
        return self.data.dtype


class Params4bit(FakeParam):
    pass


class FakeModel:
    def __init__(self, params):
        self.params = params

    def named_parameters(self):
        return iter(self.params.items())


def test_t4_converts_only_trainable_half_adapters_to_fp32():
    params = {
        "lora_a": FakeParam(FakeTorch.bfloat16),
        "lora_b": FakeParam(FakeTorch.float16),
        "lora_c": FakeParam(FakeTorch.float32),
        "frozen_base": FakeParam(FakeTorch.bfloat16, requires_grad=False),
    }
    result = normalize_t4_trainable_precision(FakeModel(params), FakeTorch, capability=(7, 5))
    assert result == {"converted": 2, "checked": 3, "skipped_newer_gpu": False}
    assert [params[x].dtype for x in ("lora_a", "lora_b", "lora_c")] == [FakeTorch.float32] * 3
    assert params["frozen_base"].dtype == FakeTorch.bfloat16
    again = normalize_t4_trainable_precision(FakeModel(params), FakeTorch, capability=(7, 5))
    assert again["converted"] == 0


def test_t4_fail_closed_on_unknown_precision_or_no_trainables():
    import pytest
    with pytest.raises(RuntimeError, match="Unsupported T4 trainable"):
        normalize_t4_trainable_precision(FakeModel({"adapter": FakeParam("float8")}), FakeTorch, capability=(7, 5))
    with pytest.raises(RuntimeError, match="No trainable"):
        normalize_t4_trainable_precision(FakeModel({"base": FakeParam("float16", requires_grad=False)}), FakeTorch, capability=(7, 5))
    with pytest.raises(RuntimeError, match="Trainable quantized"):
        normalize_t4_trainable_precision(FakeModel({"bnb": Params4bit("float16")}), FakeTorch, capability=(7, 5))


def test_newer_gpu_does_not_change_adapter_dtype():
    p = FakeParam(FakeTorch.bfloat16)
    result = normalize_t4_trainable_precision(FakeModel({"adapter": p}), FakeTorch, capability=(8, 0))
    assert result["skipped_newer_gpu"] is True
    assert p.dtype == FakeTorch.bfloat16

def test_trl_post_init_recast_is_restored_then_idempotent():
    # Reproduces physical run 44035083621: initial LoRA FP32, TRL
    # internally recasts a subset to BF16 on SFTTrainer construction.
    params = {
        "lora_A": FakeParam(FakeTorch.float32),
        "lora_B": FakeParam(FakeTorch.float32),
        "frozen_base": FakeParam(FakeTorch.bfloat16, requires_grad=False),
    }
    model = FakeModel(params)
    first = normalize_t4_trainable_precision(model, FakeTorch, capability=(7, 5))
    assert first["converted"] == 0 and first["checked"] == 2
    params["lora_B"].data = FakeTensor(FakeTorch.bfloat16)
    repaired = normalize_t4_trainable_precision(model, FakeTorch, capability=(7, 5))
    again = normalize_t4_trainable_precision(model, FakeTorch, capability=(7, 5))
    assert repaired["converted"] == 1 and repaired["checked"] == 2
    assert again["converted"] == 0 and again["checked"] == 2
    assert params["lora_A"].dtype == params["lora_B"].dtype == FakeTorch.float32
    assert params["frozen_base"].dtype == FakeTorch.bfloat16
