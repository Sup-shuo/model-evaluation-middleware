import unittest
from unittest.mock import patch

from model_evaluation.sdk.openai_image import require_chat_image
from model_evaluation.sdk.runtime import AdapterError


class OpenAIImageTests(unittest.TestCase):
    def test_image_request_uses_inline_png(self):
        with patch("model_evaluation.sdk.openai_image.request_json", return_value=(
            200, {"choices": [{"message": {"content": "red"}}]},
        )) as request:
            require_chat_image("http://localhost/v1/chat/completions", "vl")
        args = request.call_args.kwargs
        self.assertEqual(args["method"], "POST")
        self.assertTrue(args["payload"]["messages"][0]["content"][1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        ))

    def test_image_request_requires_generated_response(self):
        for reply in ({}, {"choices": [{"message": {"content": ""}}]}):
            with self.subTest(reply=reply):
                with patch("model_evaluation.sdk.openai_image.request_json", return_value=(200, reply)):
                    with self.assertRaisesRegex(AdapterError, "Image input preflight"):
                        require_chat_image("http://localhost/v1/chat/completions", "vl")

    def test_image_request_rejection_is_a_compatibility_failure(self):
        with patch("model_evaluation.sdk.openai_image.request_json", side_effect=AdapterError(
            "SERVICE_NOT_READY", "HTTP 400: image inputs not supported",
        )):
            with self.assertRaisesRegex(AdapterError, "image inputs not supported"):
                require_chat_image("http://localhost/v1/chat/completions", "vl")
