from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mindscopex_analysis.models import (
    DEFAULT_ANALYSIS_PROFILE_KEY,
    DEFAULT_BLOCK_PATH_TEMPLATE,
    DEFAULT_MODEL_ID,
    DEFAULT_QWEN_CHAT_MODEL_IDS,
    DEFAULT_QWEN_SCOPE_REPO_ID,
    QWEN35_ANALYSIS_PROFILES,
    get_qwen35_analysis_profile,
    load_qwen_language_model,
    load_qwen_text_generation_model,
)


class _LoadedModel:
    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True


class _AutoConfig:
    model_type = "qwen3_5"

    @classmethod
    def from_pretrained(cls, _model_id, **_kwargs):
        return SimpleNamespace(model_type=cls.model_type)


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
    def setUp(self) -> None:
        _AutoConfig.model_type = "qwen3_5"
        _AutoProcessor.loaded_model_id = None
        _AutoModelForMultimodalLM.loaded_model_id = None
        _AutoModelForMultimodalLM.loaded_kwargs = None

    def test_profiles_cover_requested_qwen35_family(self) -> None:
        self.assertEqual(DEFAULT_ANALYSIS_PROFILE_KEY, "27b")
        self.assertEqual(
            DEFAULT_QWEN_CHAT_MODEL_IDS,
            (
                "Qwen/Qwen3.5-2B",
                "Qwen/Qwen3.5-9B",
                "Qwen/Qwen3.5-27B",
                "Qwen/Qwen3.5-35B-A3B",
            ),
        )
        self.assertEqual(
            {
                key: (profile.num_layers, profile.hidden_size)
                for key, profile in QWEN35_ANALYSIS_PROFILES.items()
            },
            {
                "2b": (24, 2048),
                "9b": (32, 4096),
                "27b": (64, 5120),
                "35b-a3b": (40, 2048),
            },
        )
        self.assertEqual(DEFAULT_MODEL_ID, "Qwen/Qwen3.5-27B")
        self.assertEqual(DEFAULT_QWEN_SCOPE_REPO_ID, "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50")
        self.assertEqual(DEFAULT_BLOCK_PATH_TEMPLATE, "model.language_model.layers.{layer}")
        self.assertTrue(get_qwen35_analysis_profile().sae_matches_behavior_model)
        self.assertEqual(get_qwen35_analysis_profile("35B").key, "35b-a3b")
        with self.assertRaises(ValueError):
            get_qwen35_analysis_profile("4b")

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

    def test_dispatches_qwen35_moe_to_multimodal_auto_class(self) -> None:
        _AutoConfig.model_type = "qwen3_5_moe"
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoConfig = _AutoConfig
        fake_transformers.AutoModelForCausalLM = object
        fake_transformers.AutoModelForMultimodalLM = _AutoModelForMultimodalLM
        fake_transformers.AutoProcessor = _AutoProcessor
        fake_transformers.AutoTokenizer = object

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            model, _ = load_qwen_text_generation_model(
                "Qwen/Qwen3.5-35B-A3B",
                dtype=None,
            )

        self.assertTrue(model.eval_called)
        self.assertEqual(_AutoModelForMultimodalLM.loaded_model_id, "Qwen/Qwen3.5-35B-A3B")

    def test_nnsight_loader_uses_multimodal_auto_class(self) -> None:
        calls = []

        class _LanguageModel:
            def __init__(self, model_id, **kwargs):
                calls.append((model_id, kwargs))

        fake_nnsight = types.ModuleType("nnsight")
        fake_nnsight.LanguageModel = _LanguageModel
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForMultimodalLM = _AutoModelForMultimodalLM

        with patch.dict(
            sys.modules,
            {"nnsight": fake_nnsight, "transformers": fake_transformers},
        ):
            load_qwen_language_model("Qwen/Qwen3.5-2B-Base", dtype=None)

        self.assertEqual(calls[0][0], "Qwen/Qwen3.5-2B-Base")
        self.assertIs(calls[0][1]["automodel"], _AutoModelForMultimodalLM)
        self.assertEqual(calls[0][1]["dispatch"], True)


if __name__ == "__main__":
    unittest.main()
