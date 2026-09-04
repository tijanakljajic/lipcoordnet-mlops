# LipCoordNet MLOps

MLOps support for a pretrained LipCoordNet visual speech recognition model. The project provides online inference through FastAPI, request monitoring, Word Error Rate (WER) tracking, error logging, and automated tests.

## Features

- FastAPI service for indexed demo-sample inference
- automatic CPU/GPU device selection
- operational monitoring: processing time and HTTP error counts
- ML monitoring: per-request and corpus-level WER
- CSV logging for successful predictions and failed requests
- automated monitoring and API tests
- three self-contained validation examples for demonstration

## Project structure

```text
lipcoordnet-mlops/
├── api.py                 # FastAPI application and endpoints
├── predictor.py           # Model inference and CTC decoding
├── monitoring.py          # WER, CSV logging and aggregate statistics
├── model.py               # LipCoordNet model architecture
├── dataset.py             # Demo-sample loading and preprocessing
├── cvtransforms.py        # Video-frame transformations
├── options.py             # Demo-data paths and padding configuration
├── test_monitoring.py     # Automated tests
├── requirements.txt       # Python dependencies
├── demo_data/             # Three validation examples
└── pretrain/              # Pretrained model checkpoint
```

Runtime files such as `prediction_log.csv`, `error_log.csv`, Python caches, and local editor settings are excluded through `.gitignore`.

## Requirements

- Python 3.10
- PyTorch 2.1.1
- Optional NVIDIA GPU with a compatible CUDA environment

## Installation

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

For PyTorch 2.1.1 with CUDA 12.1 support, install the CUDA build from the official PyTorch index:

```powershell
python -m pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

## Running the API

Start the local development server from the project root:

```powershell
python -m uvicorn api:app --reload
```

The service is then available at `http://127.0.0.1:8000`.

### Endpoints

- `GET /health` — service health check
- `GET /predict/{sample_index}` — inference for demo samples `0`, `1`, or `2`
- `GET /stats` — aggregated latency, WER, and error statistics
- `GET /docs` — interactive OpenAPI documentation

Example:

```text
http://127.0.0.1:8000/predict/2
```

## Tests

Run the automated test suite:

```powershell
python -m pytest -q test_monitoring.py
```

The current suite contains 20 tests covering WER calculation, CSV logging, aggregate statistics, API responses, validation errors, server errors, and concurrent log writes.

## Monitoring

Successful predictions are written locally to `prediction_log.csv`. Failed prediction requests are written to `error_log.csv`. These runtime logs are generated automatically and are not committed to Git.

The `/stats` endpoint reports:

- number of logged successful predictions
- average processing time in milliseconds
- corpus-level WER over logged predictions
- number of failed requests
- client-error (`4xx`) and server-error (`5xx`) counts

## Scope and limitations

This repository demonstrates the MLOps layer around an already trained model. It does not contain the original training pipeline or the full EGCLLC dataset. Inference is currently limited to three indexed demo samples whose reference transcriptions are available, which makes request-level WER monitoring possible.

## Attribution

The LipCoordNet architecture, supporting preprocessing code, and pretrained checkpoint are based on [ffeew/LipCoordNet](https://github.com/ffeew/LipCoordNet), which identifies the project as MIT-licensed.

The three demo samples are a selected subset of the [SilentSpeak/EGCLLC dataset](https://huggingface.co/datasets/SilentSpeak/EGCLLC), based on the GRID audiovisual speech corpus and published under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/). The samples retain the source preprocessing; the modification in this repository is limited to selecting a small validation subset for demonstration.
