from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mindscopex_analysis.models import load_qwen_text_generation_model


class _LoadedModel:
    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True


class _AutoConfig:
    @classmethod
    def from_pretrained(cls, _model_id, **_kwargs):
        return SimpleNamespace(model_type="qwen3_5")


class _AutoProcessor:
    loaded_model_id = None

    @classmethod
    def from_pretrained(cls, model_id, **_kwargs):
        cls.loaded_model_id = model_id
        return SimpleNamespace(name="processor")


class _AutoModelForMultimodalLM:
    loaded_model_id = None
    loaded_kwargs = None

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.loaded_model_id = model_id
        cls.loaded_kwargs = kwargs
        return _LoadedModel()


class Qwen35LoadingTests(unittest.TestCase):
    def test_dispatches_qwen35_to_multimodal_auto_class(self) -> None:
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoConfig = _AutoConfig
        fake_transformers.AutoModelForCausalLM = object
        fake_transformers.AutoModelForMultimodalLM = _AutoModelForMultimodalLM
        fake_transformers.AutoProcessor = _AutoProcessor
        fake_transformers.AutoTokenizer = object

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            model, processor = load_qwen_text_generation_model(
                "Qwen/Qwen3.5-27B",
                dtype=None,
            )

        self.assertEqual(processor.name, "processor")
        self.assertTrue(model.eval_called)
        self.assertEqual(_AutoProcessor.loaded_model_id, "Qwen/Qwen3.5-27B")
        self.assertEqual(_AutoModelForMultimodalLM.loaded_model_id, "Qwen/Qwen3.5-27B")
        self.assertEqual(_AutoModelForMultimodalLM.loaded_kwargs["device_map"], "auto")


if __name__ == "__main__":
    unittest.main()
