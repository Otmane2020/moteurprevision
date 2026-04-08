"""
Moteur de Prévision – FastAPI server
=====================================
10 forecasting models | 80/20 backtesting | MAPE ranking
Called by React, SAP/ERP or any external API.
"""
import io
import os
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.data_processor import DataProcessor
from app.forecaster import ForecastEngine

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("moteur")

app = FastAPI(
    title="Moteur de Prévision",
    description=(
        "**API de prévision avancée**\n\n"
        "- 10 modèles : Naive, SES, Holt-Winters, ARIMA, SARIMA, Theta, "
        "Linear Regression, Random Forest, XGBoost, LightGBM (+ Prophet optionnel)\n"
        "- Backtesting automatique 80 / 20\n"
        "- Classement par **MAPE** (+ MAE, RMSE, R², SMAPE, WAPE, Bias)\n"
        "- Import CSV (React), JSON, SAP/ERP\n"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ForecastEngine()
processor = DataProcessor()


# ===========================================================================
# Pydantic Schemas
# ===========================================================================

class ForecastRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(
        ...,
        example=[
            {"date": "2022-01-01", "value": 120},
            {"date": "2022-02-01", "value": 135},
        ],
    )
    date_column: str = Field("date", description="Name of the date column")
    value_column: str = Field("value", description="Name of the value/quantity column")
    horizon: int = Field(12, ge=1, le=120, description="Number of periods to forecast")
    frequency: Optional[str] = Field(
        None, description="D=daily, W=weekly, M=monthly, Q=quarterly, Y=annual"
    )
    models: List[str] = Field(["all"], description="Model names or ['all']")
    product_column: Optional[str] = Field(None, description="Group column for multi-product")


class BacktestRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_column: str = "date"
    value_column: str = "value"
    models: List[str] = ["all"]
    split_ratio: float = Field(0.8, ge=0.5, le=0.95)
    frequency: Optional[str] = None


class SAPImportRequest(BaseModel):
    sap_data: List[Dict[str, Any]] = Field(
        ...,
        description="List of SAP/ERP records",
        example=[{"BUDAT": "2022-01-31", "MENGE": 500, "MATNR": "MAT001"}],
    )
    date_field: str = Field("BUDAT", description="SAP date field name")
    quantity_field: str = Field("MENGE", description="SAP quantity field name")
    material_field: Optional[str] = Field("MATNR", description="SAP material/product field")
    horizon: int = Field(12, ge=1, le=120)
    frequency: Optional[str] = None
    models: List[str] = ["all"]


# ===========================================================================
# Routes
# ===========================================================================

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Moteur de Prévision",
        "version": "1.0.0",
        "available_models": engine.available_models(),
        "endpoints": [
            "GET  /health",
            "GET  /models",
            "POST /forecast/upload   – CSV file",
            "POST /forecast/json     – JSON body",
            "POST /forecast/sap      – SAP/ERP data",
            "POST /backtest          – 80/20 backtesting",
            "POST /export/csv        – Download best forecast as CSV",
            "GET  /docs              – Swagger UI",
        ],
    }


@app.get("/health", tags=["Info"])
async def health():
    return {
        "status": "healthy",
        "models_available": len(engine.available_models()),
        "models": engine.available_models(),
    }


@app.get("/models", tags=["Info"])
async def list_models():
    return {"count": len(engine.model_info()), "models": engine.model_info()}


# ---------------------------------------------------------------------------
# Forecast – CSV Upload (from React / ERP export)
# ---------------------------------------------------------------------------

@app.post("/forecast/upload", tags=["Forecast"])
async def forecast_from_csv(
    file: UploadFile = File(..., description="CSV file with date and value columns"),
    horizon: int = Query(12, ge=1, le=120),
    date_column: str = Query("date"),
    value_column: str = Query("value"),
    frequency: Optional[str] = Query(None),
    models: str = Query("all", description="Comma-separated model names or 'all'"),
    product_column: Optional[str] = Query(None),
):
    """
    Upload a CSV file and run all forecasting models.

    **CSV expected format:**
    ```
    date,value
    2022-01-01,120
    2022-02-01,135
    ```
    """
    content = await file.read()
    try:
        df = processor.parse_csv_bytes(content)
    except Exception as exc:
        raise HTTPException(400, f"Invalid CSV: {exc}")

    model_list = [m.strip() for m in models.split(",")] if models != "all" else ["all"]
    log.info("CSV upload: %d rows, horizon=%d, models=%s", len(df), horizon, model_list)

    try:
        result = engine.run_forecast(
            df=df,
            date_column=date_column,
            value_column=value_column,
            horizon=horizon,
            frequency=frequency,
            models=model_list,
            product_column=product_column,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log.error(traceback_str(exc))
        raise HTTPException(500, f"Forecast error: {exc}")

    return result


# ---------------------------------------------------------------------------
# Forecast – JSON body
# ---------------------------------------------------------------------------

@app.post("/forecast/json", tags=["Forecast"])
async def forecast_from_json(request: ForecastRequest):
    """
    Send time series as JSON and get forecasts from all models.
    """
    df = pd.DataFrame(request.data)
    log.info("JSON forecast: %d rows, horizon=%d", len(df), request.horizon)
    try:
        return engine.run_forecast(
            df=df,
            date_column=request.date_column,
            value_column=request.value_column,
            horizon=request.horizon,
            frequency=request.frequency,
            models=request.models,
            product_column=request.product_column,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log.error(traceback_str(exc))
        raise HTTPException(500, f"Forecast error: {exc}")


# ---------------------------------------------------------------------------
# Forecast – SAP / ERP import
# ---------------------------------------------------------------------------

@app.post("/forecast/sap", tags=["SAP/ERP"])
async def forecast_from_sap(request: SAPImportRequest):
    """
    Import data directly from SAP/ERP field names and run forecasts.

    Map your SAP fields:
    - `date_field`     → BUDAT, BLDAT, etc.
    - `quantity_field` → MENGE, LABST, etc.
    - `material_field` → MATNR, WERKS, etc. (optional, for multi-product)
    """
    df = processor.transform_sap_data(
        request.sap_data,
        request.date_field,
        request.quantity_field,
        request.material_field,
    )
    log.info("SAP import: %d rows", len(df))
    try:
        return engine.run_forecast(
            df=df,
            date_column="date",
            value_column="value",
            horizon=request.horizon,
            frequency=request.frequency,
            models=request.models,
            product_column="product" if request.material_field else None,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log.error(traceback_str(exc))
        raise HTTPException(500, f"Forecast error: {exc}")


# ---------------------------------------------------------------------------
# Backtesting – 80/20
# ---------------------------------------------------------------------------

@app.post("/backtest", tags=["Backtesting"])
async def run_backtest(request: BacktestRequest):
    """
    Run 80/20 backtesting on historical data.
    Returns MAPE, MAE, RMSE, R², SMAPE, WAPE ranked by MAPE.
    """
    df = pd.DataFrame(request.data)
    try:
        return engine.run_backtest(
            df=df,
            date_column=request.date_column,
            value_column=request.value_column,
            models=request.models,
            split_ratio=request.split_ratio,
            frequency=request.frequency,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log.error(traceback_str(exc))
        raise HTTPException(500, f"Backtest error: {exc}")


@app.post("/backtest/upload", tags=["Backtesting"])
async def backtest_from_csv(
    file: UploadFile = File(...),
    date_column: str = Query("date"),
    value_column: str = Query("value"),
    split_ratio: float = Query(0.8),
    frequency: Optional[str] = Query(None),
):
    """Upload CSV and run backtesting."""
    content = await file.read()
    try:
        df = processor.parse_csv_bytes(content)
    except Exception as exc:
        raise HTTPException(400, f"Invalid CSV: {exc}")
    try:
        return engine.run_backtest(df, date_column, value_column, ["all"], split_ratio, frequency)
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ---------------------------------------------------------------------------
# Export – best forecast as CSV
# ---------------------------------------------------------------------------

@app.post("/export/csv", tags=["Export"])
async def export_forecast_csv(request: ForecastRequest):
    """
    Run forecast and return the best model's predictions as a downloadable CSV.
    """
    df = pd.DataFrame(request.data)
    try:
        result = engine.run_forecast(
            df=df,
            date_column=request.date_column,
            value_column=request.value_column,
            horizon=request.horizon,
            frequency=request.frequency,
            models=request.models,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))

    best = result.get("best_model", "")
    forecast_data = result.get("forecasts", {}).get(best)
    if not forecast_data or isinstance(forecast_data, dict) and "error" in forecast_data:
        raise HTTPException(500, "No valid forecast to export")

    out_df = pd.DataFrame(forecast_data)
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="forecast_{best}.csv"'},
    )


@app.post("/export/csv/upload", tags=["Export"])
async def export_forecast_csv_upload(
    file: UploadFile = File(...),
    horizon: int = Query(12),
    date_column: str = Query("date"),
    value_column: str = Query("value"),
):
    """Upload CSV and download forecast results."""
    content = await file.read()
    df = processor.parse_csv_bytes(content)
    result = engine.run_forecast(df, date_column, value_column, horizon)
    best = result["best_model"]
    out_df = pd.DataFrame(result["forecasts"][best])
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="forecast_{best}.csv"'},
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def traceback_str(exc: Exception) -> str:
    import traceback

    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
