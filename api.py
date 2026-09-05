import logging
from contextlib import asynccontextmanager
from time import perf_counter

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import options as opt
from dataset import MyDataset
from model import LipCoordNet
from monitoring import get_stats, log_failed_request, log_prediction, wer_percent
from predictor import predict

WEIGHTS = (
    "pretrain/LipCoordNet_coords_loss_0.025581153109669685_"
    "wer_0.01746208431890914_cer_0.006488426950253695.pt"
)
logger = logging.getLogger("uvicorn.error")


def record_failure(request, status_code, detail):
    path = request.url.path
    if path == "/predict" or path.startswith("/predict/"):
        try:
            log_failed_request(request.method, path, status_code, detail)
        except OSError:
            # Preserve the original HTTP error even if the CSV cannot be written.
            logger.exception("Upis u error_log.csv nije uspeo.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.model = LipCoordNet()
    app.state.model.load_state_dict(
        torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    )
    app.state.model.to(app.state.device)
    app.state.model.eval()

    app.state.dataset = MyDataset(
        opt.video_path,
        opt.anno_path,
        opt.coords_path,
        opt.val_list,
        opt.vid_padding,
        opt.txt_padding,
        "test",
    )

    try:
        yield
    finally:
        del app.state.model
        del app.state.dataset


app = FastAPI(title="LipCoordNet API", lifespan=lifespan)


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(request: Request, exc: StarletteHTTPException):
    record_failure(request, exc.status_code, str(exc.detail))
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    record_failure(request, 422, "Neispravni parametri zahteva.")
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(Exception)
async def handle_server_error(request: Request, exc: Exception):
    record_failure(request, 500, f"{type(exc).__name__}: {exc}")
    logger.error(
        "Greska pri obradi %s %s", request.method, request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Doslo je do greske pri obradi zahteva."},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict/{sample_index}")
def predict_sample(sample_index: int):
    dataset = app.state.dataset
    if not 0 <= sample_index < len(dataset):
        raise HTTPException(status_code=404, detail="Primer ne postoji.")

    started_at = perf_counter()
    sample = dataset[sample_index]
    prediction, confidence_percent = predict(
        app.state.model,
        sample["vid"].to(app.state.device),
        sample["coord"].to(app.state.device),
        int(sample["vid_len"]),
    )
    reference = MyDataset.arr2txt(
        sample["txt"][: int(sample["txt_len"])].tolist(), start=1
    )
    processing_time_ms = round((perf_counter() - started_at) * 1000, 2)

    result = {
        "sample_index": sample_index,
        "prediction": prediction,
        "confidence_percent": confidence_percent,
        "reference": reference,
        "processing_time_ms": processing_time_ms,
        "wer_percent": wer_percent(prediction, reference),
    }
    log_prediction(result)
    return result


@app.get("/stats")
def stats():
    return get_stats()
