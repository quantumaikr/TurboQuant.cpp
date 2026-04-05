"""
Tests for the quantcpp Python package.

These tests verify module structure, API surface, and error handling.
Tests that require a real model file are skipped by default.
"""

import os
import sys
import tempfile
import unittest

# Ensure the package is importable from the source tree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestImport(unittest.TestCase):
    """Verify the package can be imported."""

    def test_import_quantcpp(self):
        import quantcpp
        self.assertIsNotNone(quantcpp)

    def test_version(self):
        import quantcpp
        self.assertIsInstance(quantcpp.__version__, str)
        self.assertTrue(len(quantcpp.__version__) > 0)

    def test_model_class_importable(self):
        from quantcpp import Model
        self.assertTrue(callable(Model))

    def test_load_function_importable(self):
        from quantcpp import load
        self.assertTrue(callable(load))


class TestModelAPI(unittest.TestCase):
    """Verify Model class API surface without a real model."""

    def test_missing_file_raises(self):
        from quantcpp import Model
        with self.assertRaises(FileNotFoundError):
            Model("/nonexistent/path/model.gguf")

    def test_has_ask_method(self):
        from quantcpp import Model
        self.assertTrue(hasattr(Model, "ask"))

    def test_has_generate_method(self):
        from quantcpp import Model
        self.assertTrue(hasattr(Model, "generate"))

    def test_has_close_method(self):
        from quantcpp import Model
        self.assertTrue(hasattr(Model, "close"))

    def test_context_manager_protocol(self):
        from quantcpp import Model
        self.assertTrue(hasattr(Model, "__enter__"))
        self.assertTrue(hasattr(Model, "__exit__"))

    def test_path_property(self):
        from quantcpp import Model
        self.assertTrue(isinstance(getattr(Model, "path", None), property))

    def test_repr(self):
        from quantcpp import Model
        self.assertTrue(hasattr(Model, "__repr__"))


class TestBinding(unittest.TestCase):
    """Verify the low-level binding module."""

    def test_import_binding(self):
        from quantcpp import _binding
        self.assertIsNotNone(_binding)

    def test_quant_config_struct(self):
        from quantcpp._binding import QuantConfig
        cfg = QuantConfig(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            n_threads=4,
            kv_compress=1,
        )
        self.assertAlmostEqual(cfg.temperature, 0.7, places=5)
        self.assertAlmostEqual(cfg.top_p, 0.9, places=5)
        self.assertEqual(cfg.max_tokens, 256)
        self.assertEqual(cfg.n_threads, 4)
        self.assertEqual(cfg.kv_compress, 1)

    def test_quant_config_defaults(self):
        from quantcpp._binding import QuantConfig
        cfg = QuantConfig()
        # Fields should be zero-initialized by ctypes
        self.assertEqual(cfg.max_tokens, 0)
        self.assertEqual(cfg.n_threads, 0)

    def test_callback_type(self):
        from quantcpp._binding import ON_TOKEN_CB
        # Verify we can create a callback
        calls = []

        def my_cb(text, ud):
            calls.append(text)

        cb = ON_TOKEN_CB(my_cb)
        self.assertIsNotNone(cb)

    def test_version_function_exists(self):
        from quantcpp._binding import version
        self.assertTrue(callable(version))

    def test_binding_functions_exist(self):
        from quantcpp import _binding
        for fn in ["load_model", "new_context", "generate", "ask",
                    "free_ctx", "free_model", "version", "get_lib"]:
            self.assertTrue(
                hasattr(_binding, fn),
                f"Missing function: {fn}",
            )


class TestLibraryDiscovery(unittest.TestCase):
    """Verify library search logic (without actually loading)."""

    def test_lib_name_platform(self):
        from quantcpp._binding import _lib_name
        name = _lib_name()
        if sys.platform == "darwin":
            self.assertEqual(name, "libquant.dylib")
        elif sys.platform == "win32":
            self.assertEqual(name, "quant.dll")
        else:
            self.assertEqual(name, "libquant.so")


if __name__ == "__main__":
    unittest.main()
