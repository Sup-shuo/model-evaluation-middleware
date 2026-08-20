from __future__ import annotations

import socket
import unittest

from model_evaluation.commands.render import render_explanation
from model_evaluation.commands.workflow import _explanations, _resource_report


class WorkflowTests(unittest.TestCase):
    def test_resource_preview_rejects_active_listener_without_acquiring_leases(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            report = _resource_report(
                {
                    "plans": [
                        {
                            "plan_id": "plan-one",
                            "resources": [
                                {
                                    "kind": "port",
                                    "id": str(port),
                                    "host": "127.0.0.1",
                                    "exclusive": True,
                                }
                            ],
                        }
                    ]
                }
            )
        self.assertFalse(report["ok"])
        self.assertEqual(report["runs"][0]["claims"][0]["status"], "unavailable")

    def test_explain_reports_known_failure_without_inventing_capacity(self):
        report = {
            "ok": False,
            "system": "fixture",
            "phases": {
                "validation": {"status": "ok"},
                "planning": {
                    "status": "failed",
                    "details": {
                        "preview": [
                            {
                                "model": "large-model",
                                "compatibility": "incompatible",
                                "reasons": ["backend does not support architecture"],
                            }
                        ]
                    },
                },
                "doctor": {"status": "skipped"},
                "resources": {"status": "skipped"},
            },
        }
        report["explanations"] = _explanations(report)
        rendered = render_explanation(report)
        self.assertIn("backend does not support architecture", rendered)
        self.assertIn("unknown values are not inferred", rendered)
        self.assertNotIn("80GB", rendered)


if __name__ == "__main__":
    unittest.main()
