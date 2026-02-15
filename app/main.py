import logging
import os
import time
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

# --------------- Logging ---------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("inference-api")

# --------------- Prometheus metrics ---------------
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Inference request latency",
    ["endpoint"],
)
DRIFT_COUNT = Counter(
    "drift_detected_total",
    "Total drift detection events",
)

# --------------- FastAPI app ---------------
app = FastAPI(title="Inference API", version="1.0.0")


# --------------- Models ---------------
class PredictRequest(BaseModel):
    instances: List[List[float]]


class PredictResponse(BaseModel):
    predictions: List[int]
    drift_detected: bool


# --------------- Drift detector ---------------
def drift_detector(instances: List[List[float]]) -> bool:
    """Simple rule-based drift detector.

    Returns True if any feature value exceeds a reasonable range
    for the Iris dataset (features are typically 0-8).
    """
    for row in instances:
        if any(v > 100 or v < -10 for v in row):
            return True
    return False


# --------------- Predictor ---------------
def predict(instances: List[List[float]]) -> List[int]:
    """Dummy predictor simulating Iris classification.

    Maps feature sums to one of 3 classes (0, 1, 2).
    """
    results = []
    for row in instances:
        total = sum(row)
        if total < 10:
            results.append(0)
        elif total < 15:
            results.append(1)
        else:
            results.append(2)
    return results


# --------------- Endpoints ---------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    start = time.time()
    logger.info("Received predict request with %d instances", len(req.instances))
    logger.info("Input data: %s", req.instances)

    try:
        # Drift detection
        drift = drift_detector(req.instances)
        if drift:
            DRIFT_COUNT.inc()
            logger.warning("Drift detected for input: %s", req.instances)

        # Prediction
        preds = predict(req.instances)

        logger.info("Predictions: %s, drift_detected: %s", preds, drift)
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()

        return PredictResponse(predictions=preds, drift_detected=drift)

    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        raise

    finally:
        duration = time.time() - start
        REQUEST_LATENCY.labels(endpoint="/predict").observe(duration)
