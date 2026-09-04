"""CSV monitoring for the local, single-worker LipCoordNet service."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import editdistance

BASE_DIR = Path(__file__).resolve().parent
PREDICTION_LOG = BASE_DIR / "prediction_log.csv"
ERROR_LOG = BASE_DIR / "error_log.csv"
LOG_LOCK = Lock()


def word_error_counts(prediction, reference):
    reference_words = reference.split()
    prediction_words = prediction.split()
    errors = editdistance.eval(reference_words, prediction_words)
    return int(errors), len(reference_words)


def wer_percent(prediction, reference):
    errors, reference_words = word_error_counts(prediction, reference)
    return round(100 * errors / reference_words, 2) if reference_words else None


def _append_csv(path, row):
    with LOG_LOCK:
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=row.keys())
            if needs_header:
                writer.writeheader()
            writer.writerow(row)


def log_prediction(result):
    # Keep the original columns so previously recorded rows remain compatible.
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_index": result["sample_index"],
        "prediction": result["prediction"],
        "reference": result["reference"],
        "processing_time_ms": result["processing_time_ms"],
    }
    _append_csv(PREDICTION_LOG, row)


def log_failed_request(method, path, status_code, detail):
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "path": path,
        "status_code": status_code,
        "detail": detail,
    }
    _append_csv(ERROR_LOG, row)


def _read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as log_file:
        return list(csv.DictReader(log_file))


def get_stats():
    with LOG_LOCK:
        predictions = _read_csv(PREDICTION_LOG)
        failed_requests = _read_csv(ERROR_LOG)

    count = len(predictions)
    times = [float(row["processing_time_ms"]) for row in predictions]
    total_word_errors = 0
    total_reference_words = 0
    for row in predictions:
        errors, words = word_error_counts(row["prediction"], row["reference"])
        total_word_errors += errors
        total_reference_words += words

    # Corpus WER: divide total edits by total reference words, not by requests.
    corpus_wer = (
        round(100 * total_word_errors / total_reference_words, 2)
        if total_reference_words
        else None
    )
    status_codes = [int(row["status_code"]) for row in failed_requests]

    return {
        "logged_predictions": count,
        "avg_processing_time_ms": round(sum(times) / count, 2) if count else None,
        "logged_wer_percent": corpus_wer,
        "logged_failed_requests": len(failed_requests),
        "client_errors": sum(400 <= code < 500 for code in status_codes),
        "server_errors": sum(500 <= code < 600 for code in status_codes),
    }
