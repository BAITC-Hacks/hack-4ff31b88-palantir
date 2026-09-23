"""Exercise the API agent without credentials, network access, or paid requests."""

import copy
import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from repository_agent import AgentError, MAX_API_CALLS, MAX_TOOL_CALLS, ask


class OutputItem(SimpleNamespace):
    def model_dump(self, **kwargs):
        return vars(self).copy()


def tool(name="get_run_summary", arguments=None, call_id="call_1"):
    return OutputItem(
        type="function_call", id="fc_" + call_id, name=name,
        arguments=json.dumps({} if arguments is None else arguments),
        call_id=call_id, status="completed",
    )


def response(*items, text="", status="completed"):
    return SimpleNamespace(
        output=list(items), output_text=text, status=status,
        model="test-api-model", id="resp_fake",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


class FakeResponses:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if not self.replies:
            raise AssertionError("Agent made an unexpected extra API request")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def fake_client(*replies):
    return SimpleNamespace(responses=FakeResponses(replies))


def item_dict(item):
    return item if isinstance(item, dict) else vars(item)


def tool_outputs(call):
    return [item_dict(item) for item in call["input"]
            if item_dict(item).get("type") == "function_call_output"]


class RepositoryAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_id = "artifacts/demo"
        self.run_path = self.root / self.run_id
        self.run_path.mkdir(parents=True)
        self.issue_time = "2026-01-31T00:00:00+00:00"
        summary = {
            "mode": "lightgbm", "runs": 1, "accepted_runs": 1,
            "first_issue_date": "2026-01-31", "last_issue_date": "2026-01-31",
            "issue_hour_utc": 0, "scada_offset_h": 6,
            "replaced_points": 1, "changed_points": 0,
            "february_hours": 1, "expected_february_hours": 672,
            "complete": False,
            "provenance": {"model": "lightgbm", "weather": "fixture",
                           "utc_offset_h": 6},
        }
        (self.run_path / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8")
        point = {
            "target_time": "2026-01-31T18:00:00+00:00",
            "target_time_local": "2026-02-01T00:00:00+06:00",
            "power_pred": 0.375, "issue_time": self.issue_time,
        }
        for name in ("february_latest.csv", "forecast_history.csv"):
            row = dict(point)
            if name == "forecast_history.csv":
                row.update(lead_hour=19, accepted=True)
            with (self.run_path / name).open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
        record = {
            "issue_time": self.issue_time, "mode": "lightgbm",
            "replaced_points": 1, "provenance": summary["provenance"],
            "analysis": {
                "accepted": True, "expected_points": 1, "valid_points": 1,
                "overlap_points": 0, "changed_points": 0,
                "mean_abs_revision": None, "max_abs_revision": None,
                "errors": [], "warnings": [], "changes": [],
                "valid_forecast": [point],
                "summary": {"source": "rules", "text": "Fixture check."},
            },
        }
        (self.run_path / "runs.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8")
        model_path = self.root / "model"
        model_path.mkdir()
        (model_path / "metrics_lgbm.csv").write_text(
            "model,horizon,nMAE_%,nRMSE_%,bias_%,n\n"
            "lightgbm,all,16.43,23.83,5.9,1452\n", encoding="utf-8")

    def ask(self, client, question="Покажи результаты выбранного прогона.", **kwargs):
        return ask(question, root=self.root, run_id=self.run_id,
                   client=client, **kwargs)

    def test_reads_selected_run_and_reports_only_retrieved_sources(self):
        client = fake_client(response(tool()), response(text="Принят 1 выпуск."))
        with patch.dict(os.environ, {}, clear=True):
            result = self.ask(client)
        self.assertEqual(result["text"], "Принят 1 выпуск.")
        self.assertTrue(result["tool_calls"])
        self.assertIn("summary.json", json.dumps(result["sources"]))
        self.assertNotIn("metrics_lgbm.csv", json.dumps(result["sources"]))
        self.assertIn("usage", result)
        self.assertIn("model", result)
        self.assertEqual(result["usage"], {
            "input_tokens": 20, "output_tokens": 10, "api_calls": 2})
        self.assertEqual(result["model"], "test-api-model")
        self.assertEqual(client.responses.calls[0]["tool_choice"], "required")
        self.assertEqual(client.responses.calls[1]["tool_choice"], "auto")
        factual = json.loads(tool_outputs(client.responses.calls[1])[0]["output"])
        self.assertIn("accepted_runs", json.dumps(factual))
        self.assertIn(self.run_id, json.dumps(client.responses.calls[0], ensure_ascii=False))

    def test_forecast_checks_and_metrics_are_available_as_tools(self):
        client = fake_client(response(
            tool("get_forecast", {"start": None, "end": None,
                                   "issue_time": None, "limit": 24}, "forecast"),
            tool("get_checks", {"issue_time": None, "limit": 5}, "checks"),
            tool("get_metrics", {}, "metrics")), response(text="Данные получены."))
        result = self.ask(client)
        outputs = tool_outputs(client.responses.calls[1])
        self.assertEqual({entry["call_id"] for entry in outputs},
                         {"forecast", "checks", "metrics"})
        serialized = json.dumps(outputs, ensure_ascii=False)
        self.assertIn("0.375", serialized)
        self.assertIn("16.43", serialized)
        sources = json.dumps(result["sources"])
        for filename in ("february_latest.csv", "runs.jsonl", "metrics_lgbm.csv"):
            self.assertIn(filename, sources)

    def test_round_trip_keeps_reasoning_and_disables_server_storage(self):
        reasoning = OutputItem(type="reasoning", id="rs_1", summary=[],
                               encrypted_content="opaque-test-reasoning")
        call = tool()
        client = fake_client(response(reasoning, call), response(text="Готово."))
        self.ask(client)
        for request in client.responses.calls:
            self.assertIs(request["store"], False)
            self.assertIn("reasoning.encrypted_content", request["include"])
        continued = [item_dict(item) for item in client.responses.calls[1]["input"]]
        self.assertIn(reasoning.model_dump(), continued)
        self.assertIn(call.model_dump(), continued)

    def test_forecast_tool_forwards_selected_issue_and_calendar_dates(self):
        client = fake_client(response(tool("get_forecast", {
            "start": "2026-02-01", "end": "2026-02-01",
            "issue_time": self.issue_time, "limit": 24})),
            response(text="Прогноз выбранного выпуска."))
        result = self.ask(client)
        forecast = json.loads(tool_outputs(client.responses.calls[1])[0]["output"])
        self.assertEqual(forecast["total_rows"], 1)
        self.assertEqual(forecast["rows"][0]["issue_time"], self.issue_time)
        self.assertIn("forecast_history.csv", json.dumps(result["sources"]))
        self.assertNotIn("february_latest.csv", json.dumps(result["sources"]))

    def assert_invalid_tool_then_recovers(self, invalid):
        client = fake_client(response(invalid), response(tool(call_id="valid")),
                             response(text="Проверен выбранный прогон."))
        result = self.ask(client)
        error_output = tool_outputs(client.responses.calls[1])[0]
        self.assertIn("error", json.loads(error_output["output"]))
        self.assertIn("summary.json", json.dumps(result["sources"]))
        return client, result

    def test_unknown_tool_does_not_execute_arbitrary_code(self):
        sentinel = self.root / "unexpected.txt"
        self.assert_invalid_tool_then_recovers(tool(
            "__import__('pathlib').Path.write_text",
            {"path": str(sentinel), "text": "unexpected"}))
        self.assertFalse(sentinel.exists())

    def test_tool_cannot_override_pinned_run(self):
        other_run = self.root / "artifacts" / "other"
        other_run.mkdir()
        (other_run / "summary.json").write_text(
            '{"mode":"other","runs":999,"accepted_runs":999}', encoding="utf-8")
        client, result = self.assert_invalid_tool_then_recovers(
            tool("get_run_summary", {"run_id": "artifacts/other"}))
        outputs = json.dumps(tool_outputs(client.responses.calls[-1]))
        self.assertNotIn("999", outputs)
        self.assertNotIn("artifacts/other", json.dumps(result["sources"]))

    def test_malformed_arguments_are_reported_as_tool_errors(self):
        for arguments in ("{not-json", "[]", "null", '"a string"'):
            with self.subTest(arguments=arguments):
                invalid = tool()
                invalid.arguments = arguments
                self.assert_invalid_tool_then_recovers(invalid)

    def test_tool_limits_reject_bool_out_of_range_and_extra_arguments(self):
        for arguments in (
            {"start": None, "end": None, "issue_time": None, "limit": 0},
            {"start": None, "end": None, "issue_time": None, "limit": 169},
            {"start": None, "end": None, "issue_time": None, "limit": True},
            {"start": None, "end": None, "issue_time": None, "limit": "24"},
            {"start": None, "end": None, "issue_time": None, "limit": 24,
             "path": "../../private.csv"},
        ):
            with self.subTest(arguments=arguments):
                self.assert_invalid_tool_then_recovers(tool("get_forecast", arguments))

    def test_data_errors_are_returned_to_model_without_claiming_source_success(self):
        (self.run_path / "february_latest.csv").unlink()
        client = fake_client(response(tool("get_forecast", {
            "start": None, "end": None, "issue_time": None, "limit": 24})),
            response(tool(call_id="valid")), response(text="Доступно резюме."))
        result = self.ask(client)
        output = json.loads(tool_outputs(client.responses.calls[1])[0]["output"])
        self.assertIn("error", output)
        self.assertNotIn("february_latest.csv", json.dumps(result["sources"]))

    def test_answer_without_retrieved_facts_is_not_success(self):
        with self.assertRaises(AgentError):
            self.ask(fake_client(response(text="Уверенный ответ без чтения файлов.")))
        client = fake_client(response(tool("unknown_tool")),
                             response(text="Все проверки пройдены."))
        with self.assertRaises(AgentError):
            self.ask(client)

    def test_empty_and_incomplete_final_responses_are_errors(self):
        for final in (response(), response(text="   "),
                      response(text="Ответ обрезан", status="incomplete")):
            with self.subTest(status=final.status, text=final.output_text):
                with self.assertRaises(AgentError):
                    self.ask(fake_client(response(tool()), final))

    def test_api_errors_hide_credentials_and_private_exception_text(self):
        secret = "sk-test-credential-must-never-appear"
        client = fake_client(RuntimeError(
            "Authorization: Bearer " + secret + "; raw backend diagnostic"))
        with self.assertRaises(AgentError) as caught:
            self.ask(client)
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn("raw backend diagnostic", str(caught.exception))
        self.assertTrue(str(caught.exception).strip())

    def test_api_loop_forces_final_answer_at_limit(self):
        self.assertEqual(MAX_API_CALLS, 5)
        replies = [response(tool(call_id=f"call_{i}"))
                   for i in range(MAX_API_CALLS - 1)]
        client = fake_client(*replies, response(text="Итог."))
        result = self.ask(client)
        self.assertEqual(result["text"], "Итог.")
        self.assertEqual(len(client.responses.calls), MAX_API_CALLS)
        self.assertEqual(client.responses.calls[-1]["tool_choice"], "none")

    def test_tool_call_after_final_limit_cannot_cause_another_request(self):
        client = fake_client(*[response(tool(call_id=f"call_{i}"))
                               for i in range(MAX_API_CALLS)])
        with self.assertRaises(AgentError):
            self.ask(client)
        self.assertEqual(len(client.responses.calls), MAX_API_CALLS)

    def test_excessive_tool_batch_is_rejected_before_it_reads_data(self):
        client = fake_client(response(*[
            tool(call_id=f"call_{i}") for i in range(MAX_TOOL_CALLS + 1)]))
        with patch("repository_agent.RepositoryData.run_summary") as read:
            with self.assertRaises(AgentError):
                self.ask(client)
        read.assert_not_called()
        self.assertEqual(len(client.responses.calls), 1)

    def test_supplied_model_is_used_for_every_api_request(self):
        client = fake_client(response(tool()), response(text="Готово."))
        self.ask(client, model="configured-model")
        self.assertEqual([call["model"] for call in client.responses.calls],
                         ["configured-model", "configured-model"])

    def test_history_excludes_privileged_roles_and_is_bounded(self):
        history = [
            {"role": "system", "content": "BAD_SYSTEM_OVERRIDE"},
            {"role": "developer", "content": "BAD_DEVELOPER_OVERRIDE"},
            {"role": "tool", "content": "BAD_TOOL_OVERRIDE"},
        ] + [{"role": "user" if i % 2 == 0 else "assistant",
              "content": f"HISTORY_{i}:" + "x" * 12000} for i in range(10)]
        client = fake_client(response(tool()), response(text="Готово."))
        self.ask(client, question="CURRENT_QUESTION", history=history)
        serialized = json.dumps(client.responses.calls[0]["input"], ensure_ascii=False)
        for forbidden in ("BAD_SYSTEM_OVERRIDE", "BAD_DEVELOPER_OVERRIDE",
                          "BAD_TOOL_OVERRIDE", "HISTORY_0:", "HISTORY_3:"):
            self.assertNotIn(forbidden, serialized)
        self.assertIn("HISTORY_9:", serialized)
        self.assertIn("CURRENT_QUESTION", serialized)
        history_items = [item_dict(item) for item in client.responses.calls[0]["input"]
                         if "HISTORY_" in str(item_dict(item).get("content", ""))]
        self.assertLessEqual(len(history_items), 6)
        self.assertTrue(all(len(str(item["content"])) < 12000 for item in history_items))

    def test_blank_question_is_rejected_before_api_request(self):
        client = fake_client()
        with self.assertRaises(AgentError):
            self.ask(client, question="   ")
        self.assertEqual(client.responses.calls, [])

    def test_missing_key_has_actionable_error_without_network_request(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AgentError) as caught:
                ask("Покажи прогноз", root=self.root, run_id=self.run_id)
        self.assertIn("OPENAI_API_KEY", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
