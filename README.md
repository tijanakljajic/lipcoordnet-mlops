# LipCoordNet MLOps

[![Automated tests](https://github.com/tijanakljajic/lipcoordnet-mlops/actions/workflows/tests.yml/badge.svg)](https://github.com/tijanakljajic/lipcoordnet-mlops/actions/workflows/tests.yml)

MLOps support for a pretrained LipCoordNet visual speech recognition model. The project provides FastAPI inference, prediction monitoring, model-specific metrics, automated evaluation, Metaflow-based maintenance, containerization, and continuous integration.

## Features

- FastAPI service for indexed demo-sample inference
- automatic CPU/GPU device selection
- model confidence for every prediction
- operational monitoring of processing time and HTTP errors
- ML monitoring with request-level and corpus-level Word Error Rate (WER)
- CSV logging for successful predictions and failed requests
- automated model evaluation with a maximum average WER threshold of 10%
- Metaflow maintenance pipeline for checkpoint validation and model evaluation
- Docker image for reproducible CPU deployment
- GitHub Actions workflow for tests, model evaluation, and Docker builds
- scheduled weekly checks and manual workflow execution
- 20 automated tests
- three self-contained validation examples for demonstration

## Project structure

| Path | Purpose |
|---|---|
| `api.py` | FastAPI application and endpoints |
| `predictor.py` | Model inference, CTC decoding, and confidence calculation |
| `monitoring.py` | WER calculation, CSV logging, and aggregate statistics |
| `evaluate_model.py` | Evaluation of all demo samples and WER quality gate |
| `maintenance_flow.py` | Metaflow checkpoint-validation and evaluation pipeline |
| `model.py` | LipCoordNet model architecture |
| `dataset.py` | Demo-sample loading and preprocessing |
| `cvtransforms.py` | Video-frame transformations |
| `options.py` | Demo-data paths and padding configuration |
| `test_monitoring.py` | Automated monitoring and API tests |
| `Dockerfile` | Reproducible CPU container definition |
| `.github/workflows/tests.yml` | Continuous integration and scheduled maintenance |
| `requirements.txt` | Pinned Python dependencies |
| `demo_data/` | Three validation examples |
| `pretrain/` | Pretrained model checkpoint |

Runtime files such as `prediction_log.csv`, `error_log.csv`, Metaflow metadata, Python caches, and local editor settings are excluded from Git.

## Requirements

- Python 3.10
- PyTorch 2.1.1
- Docker Desktop for container execution
- optional NVIDIA GPU with a compatible CUDA environment for local inference

## Installation

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade `pip`, optionally install the CUDA 12.1 build of PyTorch, and install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The Docker image uses the CPU build of PyTorch so it can run without an NVIDIA runtime.

## Running the API locally

Start the development server from the project root:

```powershell
python -m uvicorn api:app --reload
```

The service is available at `http://127.0.0.1:8000`.

### Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Service health check |
| `GET /predict/{sample_index}` | Inference for demo sample `0`, `1`, or `2` |
| `GET /stats` | Aggregate latency, confidence, WER, and error statistics |
| `GET /docs` | Interactive OpenAPI documentation |

Example prediction response:

```json
{
  "sample_index": 2,
  "prediction": "LAY BLUE WITH E SIX PLEASE",
  "confidence_percent": 97.59,
  "reference": "LAY BLUE WITH E SIX PLEASE",
  "processing_time_ms": 276.49,
  "wer_percent": 0.0
}
```

Processing time depends on the available hardware.

## Monitoring

Successful predictions are written to `prediction_log.csv`. Failed prediction requests are written to `error_log.csv`. These runtime logs are generated automatically and are not committed to Git.

The `/stats` endpoint reports:

- number of logged successful predictions
- average processing time in milliseconds
- average confidence over predictions that contain a confidence value
- corpus-level WER over logged predictions
- number of failed requests
- client-error (`4xx`) and server-error (`5xx`) counts

### Confidence

The model returns raw logits. The confidence metric is calculated by applying softmax, selecting the highest class probability at each valid time step, excluding CTC blank tokens, and averaging the remaining probabilities. The value is exposed as `confidence_percent` and logged for aggregate monitoring.

This is an approximate, uncalibrated confidence score. A high score does not guarantee a correct prediction. For example, the model reports high confidence for demo sample `0` even though one word is incorrect.

## Automated model evaluation

Run the evaluation locally:

```powershell
python evaluate_model.py
```

The script evaluates all three demo samples, reports prediction, reference, confidence, and WER for each sample, and calculates average WER. The current average WER is `5.56%`.

The process exits with an error when average WER exceeds the configured `10%` threshold. This turns evaluation into a model-quality gate for automated workflows.

## Metaflow maintenance pipeline

The maintenance flow contains two substantive ML maintenance tasks:

1. `validate_checkpoint` verifies that exactly one non-empty model checkpoint is available and records its path and size.
2. `evaluate_model` runs the automated evaluation and stops the flow if the model fails the WER quality gate.

Metaflow also provides the required `start` and `end` control steps and records the status and artifacts of each run.

Run the flow directly on Linux:

```bash
METAFLOW_USER=user python maintenance_flow.py run
```

Metaflow does not run natively in the Windows Python environment used by this project because it depends on Unix functionality. On Windows, run it through the Docker Linux environment:

```powershell
New-Item -ItemType Directory -Force .metaflow | Out-Null
docker run --rm -e METAFLOW_USER=tijana -v "${PWD}/.metaflow:/app/.metaflow" lipcoordnet-api:1.1 python maintenance_flow.py run
```

## Docker

Build the CPU image:

```powershell
docker build -t lipcoordnet-api:1.1 .
```

Run the API container:

```powershell
docker run --rm -p 8001:8000 lipcoordnet-api:1.1
```

The containerized service is available at:

- `http://127.0.0.1:8001/health`
- `http://127.0.0.1:8001/docs`
- `http://127.0.0.1:8001/predict/2`

## Tests

Run the automated test suite:

```powershell
python -m pytest -q test_monitoring.py
```

The suite contains 20 tests covering WER calculation, CSV schema migration and logging, aggregate statistics, confidence propagation, API responses, validation errors, server errors, and concurrent log writes.

## Continuous integration and scheduled maintenance

The GitHub Actions workflow runs on:

- every push
- every pull request
- manual execution through **Run workflow**
- every Monday at 09:00 in the `Europe/Belgrade` time zone

The workflow performs the following stages:

1. installs pinned dependencies
2. runs the automated test suite
3. runs the Metaflow checkpoint-validation and model-evaluation pipeline
4. builds the Docker image

A failed test, invalid checkpoint, average WER above 10%, or failed Docker build causes the workflow to fail.

## Scope and limitations

This repository demonstrates the MLOps layer around an already trained model. It does not contain the original training pipeline or the full EGCLLC dataset. The maintenance pipeline validates and evaluates a candidate pretrained checkpoint instead of retraining the model.

Inference is limited to three indexed demo samples whose reference transcriptions are available. This enables demonstration of request-level WER but is not a substitute for evaluation on a full validation or production dataset. The confidence score is an uncalibrated approximation, and the CSV monitoring implementation is intended for a local, single-worker demonstration rather than a distributed production deployment.

## Attribution

The LipCoordNet architecture, supporting preprocessing code, and pretrained checkpoint are based on [ffeew/LipCoordNet](https://github.com/ffeew/LipCoordNet), which identifies the project as MIT-licensed.

The three demo samples are a selected subset of the [SilentSpeak/EGCLLC dataset](https://huggingface.co/datasets/SilentSpeak/EGCLLC), based on the GRID audiovisual speech corpus and published under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/). The samples retain the source preprocessing; the modification in this repository is limited to selecting a small validation subset for demonstration.
