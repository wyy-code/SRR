"""Attention-runtime selection must never silently change paper defaults."""

import os
from unittest import TestCase
from unittest.mock import patch

from srr.modeling import attention_implementation


class AttentionRuntimeTests(TestCase):
    def test_explicit_portability_backend(self):
        with patch.dict(os.environ, {"SRR_ATTN_IMPLEMENTATION": "sdpa"}):
            self.assertEqual(attention_implementation(), "sdpa")
        self.assertEqual(attention_implementation("eager"), "eager")

    def test_invalid_backend_rejected(self):
        with self.assertRaises(ValueError):
            attention_implementation("auto")

    def test_default_requires_flash_attention(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("srr.modeling.importlib.util.find_spec", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "FlashAttention 2 is not installed"):
                    attention_implementation()
