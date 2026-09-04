"""Test metrics, CSV compatibility and HTTP behavior without model weights.

Run from this folder: python -m pytest -q test_monitoring.py
The HTTP tests substitute the model/data; they do not test LipCoordNet accuracy.
"""

import csv
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

import monitoring


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring, "PREDICTION_LOG", tmp_path / "predictions.csv")
    monkeypatch.setattr(monitoring, "ERROR_LOG", tmp_path / "errors.csv")
    monkeypatch.setattr(monitoring, "LOG_LOCK", Lock())


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


@pytest.mark.parametrize("prediction,reference,expected", [
    ("LAY BLUE WITH E SIX PLEASE", "LAY BLUE WITH E SIX PLEASE", 0.0),
    ("PLACE GREEN WITH P EIGHT NOW", "PLACE GREEN WITH Y EIGHT NOW", 16.67),
    ("HELLO", "HELLO WORLD", 50.0),
    ("A B C", "A", 200.0),
    ("HELLO  WORLD", "HELLO WORLD", 0.0),
    ("", "", None),
    ("HELLO", "", None),
])
def test_wer_examples(prediction, reference, expected):
    assert monitoring.wer_percent(prediction, reference) == expected


def test_empty_stats():
    assert monitoring.get_stats() == {
        "logged_predictions": 0,
        "avg_processing_time_ms": None,
        "logged_wer_percent": None,
        "logged_failed_requests": 0,
        "client_errors": 0,
        "server_errors": 0,
    }


def test_legacy_csv_is_preserved_and_corpus_wer_is_weighted():
    original = (
        "timestamp_utc,sample_index,prediction,reference,processing_time_ms\n"
        "2026-09-03T13:00:00+00:00,0,PLACE GREEN WITH P EIGHT NOW,"
        "PLACE GREEN WITH Y EIGHT NOW,100\n"
    )
    monitoring.PREDICTION_LOG.write_text(original, encoding="utf-8")
    before = monitoring.PREDICTION_LOG.read_bytes()
    monitoring.log_prediction({
        "sample_index": 1, "prediction": "HELLO WORLD",
        "reference": "HELLO WORLD", "processing_time_ms": 300, "wer_percent": 0,
    })
    assert monitoring.PREDICTION_LOG.read_bytes().startswith(before)
    rows = read_rows(monitoring.PREDICTION_LOG)
    assert len(rows) == 2
    assert all(len(row) == 5 and None not in row for row in rows)
    stats = monitoring.get_stats()
    assert stats["logged_predictions"] == 2
    assert stats["avg_processing_time_ms"] == 200.0
    assert stats["logged_wer_percent"] == 12.5  # One error / eight words.


def test_concurrent_append_keeps_rows_and_one_header():
    def write(index):
        monitoring.log_prediction({
            "sample_index": index, "prediction": 'A, "B" Č',
            "reference": 'A, "B" Č', "processing_time_ms": 10,
        })

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(16)))
    rows = read_rows(monitoring.PREDICTION_LOG)
    assert len(rows) == 16
    assert {int(row["sample_index"]) for row in rows} == set(range(16))
    assert all(row["prediction"] == 'A, "B" Č' for row in rows)


@pytest.fixture
def api_client(monkeypatch):
    class Tensor:
        def __init__(self, value):
            self.value = value

        def to(self, device):
            return self

        def __getitem__(self, item):
            return Tensor(self.value[item])

        def tolist(self):
            return self.value

    class Dataset:
        @staticmethod
        def arr2txt(tokens, start):
            return " ".join(tokens)

    calls = []

    def fake_predict(model, video, coords, length):
        calls.append(video.value)
        return video.value

    replacements = {
        "torch": {}, "options": {}, "dataset": {"MyDataset": Dataset},
        "model": {"LipCoordNet": object}, "predictor": {"predict": fake_predict},
    }
    for name, values in replacements.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "api_under_test", Path(__file__).with_name("api.py")
    )
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    api.app.state.model = object()
    api.app.state.device = "cpu"
    reference = "PLACE GREEN WITH Y EIGHT NOW".split()
    api.app.state.dataset = [{
        "vid": Tensor("PLACE GREEN WITH P EIGHT NOW"),
        "coord": Tensor([]), "vid_len": 6,
        "txt": Tensor(reference), "txt_len": len(reference),
    }]
    # No context manager: the real lifespan/model loading is intentionally skipped.
    client = TestClient(api.app, raise_server_exceptions=False)
    try:
        yield api, client, calls
    finally:
        client.close()


def test_success_response_and_stats(api_client):
    api, client, calls = api_client
    response = client.get("/predict/0")
    assert response.status_code == 200
    assert response.json()["wer_percent"] == 16.67
    assert response.json()["processing_time_ms"] >= 0
    assert len(calls) == 1
    stats = client.get("/stats").json()
    assert stats["logged_predictions"] == 1
    assert stats["logged_wer_percent"] == 16.67
    assert stats["logged_failed_requests"] == 0


@pytest.mark.parametrize("path,status", [
    ("/predict/-1", 404), ("/predict/1", 404),
    ("/predict/abc", 422), ("/predict", 404),
])
def test_bad_prediction_requests_are_logged_once(api_client, path, status):
    api, client, calls = api_client
    response = client.get(path)
    assert response.status_code == status
    if path == "/predict/-1":
        assert response.json() == {"detail": "Primer ne postoji."}
    assert not calls
    rows = read_rows(monitoring.ERROR_LOG)
    assert len(rows) == 1
    assert int(rows[0]["status_code"]) == status
    stats = client.get("/stats").json()
    assert stats["logged_predictions"] == 0
    assert stats["logged_failed_requests"] == stats["client_errors"] == 1
    assert stats["server_errors"] == 0


def test_server_failure_is_logged_without_success_row(api_client, monkeypatch):
    api, client, calls = api_client

    def fail(*args):
        raise RuntimeError("Synthetic inference failure")

    monkeypatch.setattr(api, "predict", fail)
    response = client.get("/predict/0")
    assert response.status_code == 500
    assert response.json() == {"detail": "Doslo je do greske pri obradi zahteva."}
    rows = read_rows(monitoring.ERROR_LOG)
    assert len(rows) == 1
    assert "RuntimeError" in rows[0]["detail"]
    stats = client.get("/stats").json()
    assert stats["logged_predictions"] == 0
    assert stats["server_errors"] == stats["logged_failed_requests"] == 1


def test_csv_write_failure_is_a_logged_server_error(api_client, monkeypatch):
    api, client, calls = api_client

    def fail(result):
        raise PermissionError("Synthetic locked prediction log")

    monkeypatch.setattr(api, "log_prediction", fail)
    assert client.get("/predict/0").status_code == 500
    assert monitoring.get_stats()["server_errors"] == 1
    assert monitoring.get_stats()["logged_predictions"] == 0


def test_error_log_failure_preserves_original_http_status(api_client, monkeypatch):
    api, client, calls = api_client

    def fail(*args):
        raise PermissionError("Synthetic locked error log")

    monkeypatch.setattr(api, "log_failed_request", fail)
    response = client.get("/predict/-1")
    assert response.status_code == 404
    assert response.json() == {"detail": "Primer ne postoji."}


def test_docs_health_and_stats_do_not_count_as_prediction_failures(api_client):
    api, client, calls = api_client
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/favicon.ico").status_code == 404
    assert client.get("/stats").json()["logged_failed_requests"] == 0
    assert not calls


def test_method_error_preserves_allow_header(api_client):
    api, client, calls = api_client
    response = client.post("/predict/0")
    assert response.status_code == 405
    assert "GET" in response.headers["allow"]
    assert monitoring.get_stats()["client_errors"] == 1
    assert not calls
