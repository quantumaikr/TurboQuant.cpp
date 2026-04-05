"""
Tests for the quantcpp Python bindings.

These tests verify that the module structure and API surface are correct.
Tests that require a model file are skipped if no model is available.
"""

import importlib
import os
import sys
import unittest


class TestModuleStructure(unittest.TestCase):
    """Verify the quantcpp module can be imported and has expected API."""

    def test_import(self):
        """Module imports without error."""
        import quantcpp
        self.assertIsNotNone(quantcpp)

    def test_version(self):
        """Module exposes __version__."""
        import quantcpp
        self.assertEqual(quantcpp.__version__, "0.5.0")

    def test_model_class_exists(self):
        """Model class is exported."""
        from quantcpp import Model
        self.assertTrue(callable(Model))

    def test_load_function_exists(self):
        """load() convenience function is exported."""
        from quantcpp import load
        self.assertTrue(callable(load))

    def test_kv_type_constants(self):
        """KV compression type constants are defined."""
        import quantcpp
        self.assertEqual(quantcpp.TQ_TYPE_TURBO_KV_3B, 8)
        self.assertEqual(quantcpp.TQ_TYPE_TURBO_KV_4B, 9)
        self.assertEqual(quantcpp.TQ_TYPE_TURBO_KV_1B, 10)
        self.assertEqual(quantcpp.TQ_TYPE_TURBO_KV_2B, 11)

    def test_kv_compress_map(self):
        """KV compression string map covers expected keys."""
        from quantcpp import _KV_COMPRESS_MAP
        for key in ["1bit", "2bit", "3bit", "4bit", "polar3", "polar4",
                     "qjl", "turbo3", "turbo4", "uniform2", "uniform3",
                     "uniform4", "none"]:
            self.assertIn(key, _KV_COMPRESS_MAP, f"Missing key: {key}")


class TestModelAPI(unittest.TestCase):
    """Verify Model class API surface (no actual model needed)."""

    def test_model_init_missing_file(self):
        """Model() raises FileNotFoundError for missing path."""
        from quantcpp import Model
        with self.assertRaises(FileNotFoundError):
            Model("/nonexistent/model.gguf")

    def test_model_invalid_kv_compress(self):
        """Model() raises ValueError for invalid kv_compress."""
        from quantcpp import Model
        # Create a dummy file to pass the file-exists check
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
            with self.assertRaises(ValueError):
                Model(f.name, kv_compress="invalid_mode")

    def test_model_has_methods(self):
        """Model class exposes expected methods."""
        from quantcpp import Model
        for method in ["ask", "chat", "generate", "complete", "close"]:
            self.assertTrue(hasattr(Model, method), f"Missing method: {method}")

    def test_model_has_properties(self):
        """Model class exposes expected properties."""
        from quantcpp import Model
        for prop in ["path", "kv_type"]:
            self.assertTrue(
                isinstance(getattr(Model, prop, None), property),
                f"Missing property: {prop}",
            )

    def test_model_context_manager(self):
        """Model supports context manager protocol."""
        from quantcpp import Model
        self.assertTrue(hasattr(Model, "__enter__"))
        self.assertTrue(hasattr(Model, "__exit__"))


class TestChatTemplate(unittest.TestCase):
    """Verify chat template wrapping logic."""

    def test_qwen_template(self):
        """Qwen/ChatML template for model_type=0."""
        from quantcpp import _chat_wrap
        result = _chat_wrap("Hello", model_type=0)
        self.assertIn("<|im_start|>user", result)
        self.assertIn("Hello", result)
        self.assertIn("<|im_start|>assistant", result)

    def test_gemma_template(self):
        """Gemma template for model_type=1."""
        from quantcpp import _chat_wrap
        result = _chat_wrap("Hello", model_type=1)
        self.assertIn("<start_of_turn>user", result)
        self.assertIn("Hello", result)
        self.assertIn("<start_of_turn>model", result)

    def test_default_template(self):
        """Default template (model_type=2) uses ChatML."""
        from quantcpp import _chat_wrap
        result = _chat_wrap("Hello", model_type=2)
        self.assertIn("<|im_start|>", result)


class TestGenConfig(unittest.TestCase):
    """Verify generation config struct construction."""

    def test_make_gen_config(self):
        """_make_gen_config returns struct with expected fields."""
        from quantcpp import _make_gen_config, TQ_TYPE_TURBO_KV_4B
        cfg = _make_gen_config(
            kv_type=TQ_TYPE_TURBO_KV_4B,
            temperature=0.8,
            top_p=0.95,
            max_tokens=256,
        )
        self.assertAlmostEqual(cfg.temperature, 0.8, places=5)
        self.assertAlmostEqual(cfg.top_p, 0.95, places=5)
        self.assertEqual(cfg.max_tokens, 256)
        self.assertEqual(cfg.kv_type, TQ_TYPE_TURBO_KV_4B)


if __name__ == "__main__":
    unittest.main()
