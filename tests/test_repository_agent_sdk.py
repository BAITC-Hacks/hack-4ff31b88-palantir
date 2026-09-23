"""One real-SDK contract test, using an HTTP transport with no network access."""

import json
from pathlib import Path
import tempfile
import unittest

try:
    from openai import OpenAI
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    OpenAI = None
else:
    import httpx

from repository_agent import ask


@unittest.skipIf(OpenAI is None, "Optional OpenAI SDK is not installed")
class RepositoryAgentSDKTests(unittest.TestCase):
    def test_sdk_serializes_and_parses_responses_tool_round_trip(self):
        requests = []
        reasoning = {
            "id": "rs_mock", "type": "reasoning", "summary": [],
            "encrypted_content": "opaque-mock-content",
        }
        function_call = {
            "id": "fc_mock", "type": "function_call", "status": "completed",
            "name": "get_run_summary", "arguments": "{}", "call_id": "call_mock",
        }
        final_text = "Принят 1 выпуск. Это результат проверки, а не оценка точности."

        def transport(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/v1/responses")
            body = json.loads(request.content)
            requests.append(body)
            self.assertIs(body["store"], False)
            self.assertIn("reasoning.encrypted_content", body["include"])
            self.assertEqual(body["model"], "sdk-test-model")
            if len(requests) == 1:
                self.assertEqual(body["tool_choice"], "required")
                output = [reasoning, function_call]
            elif len(requests) == 2:
                self.assertEqual(body["tool_choice"], "auto")
                self.assertIn(reasoning, body["input"])
                self.assertIn(function_call, body["input"])
                tool_results = [item for item in body["input"]
                                if item.get("type") == "function_call_output"]
                self.assertEqual(len(tool_results), 1)
                self.assertEqual(tool_results[0]["call_id"], "call_mock")
                facts = json.loads(tool_results[0]["output"])
                self.assertEqual(facts["summary"]["accepted_runs"], 1)
                self.assertIn("artifacts/demo/summary.json", facts["sources"])
                output = [{
                    "id": "msg_mock", "type": "message", "status": "completed",
                    "role": "assistant", "content": [{
                        "type": "output_text", "text": final_text,
                        "annotations": [], "logprobs": [],
                    }],
                }]
            else:
                self.fail("Unexpected extra API request")
            return httpx.Response(200, json={
                "id": f"resp_mock_{len(requests)}", "object": "response",
                "created_at": 1770000000, "status": "completed",
                "error": None, "incomplete_details": None,
                "instructions": body["instructions"], "model": "sdk-test-model",
                "output": output, "parallel_tool_calls": False,
                "tool_choice": body["tool_choice"], "tools": body["tools"],
                "temperature": 1.0, "top_p": 1.0, "metadata": {},
                "usage": {
                    "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            })

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "artifacts" / "demo"
            run.mkdir(parents=True)
            (run / "summary.json").write_text(json.dumps({
                "mode": "lightgbm", "runs": 1, "accepted_runs": 1,
                "scada_offset_h": 6, "complete": False,
            }), encoding="utf-8")
            (run / "runs.jsonl").write_text(json.dumps({
                "issue_time": "2026-01-31T00:00:00+00:00",
                "analysis": {
                    "accepted": True, "errors": [], "warnings": [],
                    "overlap_points": 0, "changed_points": 0,
                    "mean_abs_revision": None, "max_abs_revision": None,
                    "summary": {"source": "rules", "text": "Offline fixture."},
                },
            }) + "\n", encoding="utf-8")
            with OpenAI(
                api_key="sk-offline-test-only",
                base_url="https://api.openai.com/v1",
                max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(transport)),
            ) as client:
                result = ask("Покажи результаты проверки.", root=root,
                             run_id="artifacts/demo", model="sdk-test-model", client=client)

        self.assertEqual(len(requests), 2)
        self.assertEqual(result["text"], final_text)
        self.assertEqual(result["response_id"], "resp_mock_2")
        self.assertEqual(result["model"], "sdk-test-model")
        self.assertEqual(result["sources"], [
            "artifacts/demo/runs.jsonl", "artifacts/demo/summary.json"])
        self.assertEqual(result["tool_calls"][0]["name"], "get_run_summary")
        self.assertEqual(result["usage"], {
            "input_tokens": 20, "output_tokens": 10, "api_calls": 2})


if __name__ == "__main__":
    unittest.main()
