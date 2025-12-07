import io
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.api.inference_service import SegmentationService
from src.api.schemas import ErrorResponse, HealthResponse, ModelMetadata, PredictionSummary
from src.api.settings import Settings

logger = logging.getLogger("unet3d.api")
logging.basicConfig(level=logging.INFO)

settings = Settings()
service = SegmentationService(settings=settings)


def get_service() -> SegmentationService:
    return service


def _ensure_nifti(filename: str) -> None:
    allowed = (".nii", ".nii.gz")
    if not filename or not any(filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato no soportado. Usa archivos NIfTI ({', '.join(allowed)}).",
        )


app = FastAPI(
    title="UNet3D Segmentation API",
    description=(
        "API para inferencia y monitoreo del modelo UNet3D entrenado para segmentación volumétrica. "
        "Permite subir volúmenes NIfTI y descargar la máscara predicha en el mismo formato."
    ),
    version="1.0.0",
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    openapi_url=settings.openapi_url,
    contact={
        "name": "Equipo MLE",
        "email": "ml-team@example.com",
    },
    license_info={"name": "Proprietary"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _load_on_startup() -> None:
    try:
        model = service.load_model()
        logger.info("Modelo cargado en %s", service.device)
        logger.info("Pesos: %s", settings.model_path)
        logger.info("Salidas: %s clases", getattr(model, "out_conv").out_channels)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("No se pudo cargar el modelo en startup: %s", exc)


@app.get(
    "/",
    summary="Ping",
    response_model=HealthResponse,
    tags=["monitoring"],
)
async def root(service: Annotated[SegmentationService, Depends(get_service)]) -> HealthResponse:
    """Ping simple con estado del modelo."""
    return HealthResponse(
        status="ok",
        device=str(service.device),
        model_loaded=service.model_ready,
        model_path=str(settings.model_path),
        detail=None if service.model_ready else "Modelo no cargado aún.",
    )


@app.get(
    "/health",
    summary="Estado de la API",
    response_model=HealthResponse,
    tags=["monitoring"],
)
async def health(service: Annotated[SegmentationService, Depends(get_service)]) -> HealthResponse:
    detail = None
    if not service.model_ready:
        detail = "Modelo no cargado; revisa UNET3D_MODEL_PATH o logs de arranque."
    return HealthResponse(
        status="ok",
        device=str(service.device),
        model_loaded=service.model_ready,
        model_path=str(settings.model_path),
        detail=detail,
    )


@app.get(
    "/model/metadata",
    summary="Metadatos del modelo",
    response_model=ModelMetadata,
    tags=["model"],
)
async def model_metadata(service: Annotated[SegmentationService, Depends(get_service)]) -> ModelMetadata:
    return ModelMetadata(
        model_path=str(settings.model_path),
        num_classes=settings.num_classes,
        class_names=settings.class_names,
        in_channels=settings.in_channels,
        base_channels=settings.base_channels,
        norm=settings.norm,
        dropout=settings.dropout,
        pad_multiple=settings.pad_multiple,
        clip_percentiles=settings.clip_percentiles,
        default_threshold=settings.default_threshold,
        device=str(service.device),
    )


@app.post(
    "/v1/predict",
    summary="Segmenta un volumen NIfTI",
    responses={
        200: {"model": PredictionSummary, "description": "Respuesta en JSON con metadatos de la máscara."},
        201: {"content": {"application/octet-stream": {}}, "description": "Máscara NIfTI para descargar."},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["inference"],
)
async def predict(
    file: UploadFile = File(..., description="Archivo NIfTI (.nii o .nii.gz)"),
    return_binary: bool = Query(
        False,
        description="Devuelve la máscara como archivo NIfTI (.nii.gz) en lugar de JSON.",
    ),
    threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Umbral para segmentación binaria. Si None usa el valor por defecto."
    ),
    service: Annotated[SegmentationService, Depends(get_service)],
):
    _ensure_nifti(file.filename)

    suffix = ".nii.gz" if file.filename and file.filename.lower().endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copyfileobj(file.file, tmp)

    try:
        result = service.predict(tmp_path, threshold=threshold)
    except FileNotFoundError as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        tmp_path.unlink(missing_ok=True)
        logger.exception("Fallo en inferencia: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    if return_binary:
        headers = {"Content-Disposition": f'attachment; filename="{result.filename}"'}
        return StreamingResponse(
            io.BytesIO(result.mask_bytes),
            media_type="application/octet-stream",
            headers=headers,
            status_code=status.HTTP_201_CREATED,
        )

    summary = PredictionSummary(
        filename=result.filename,
        input_shape=result.input_shape,
        padded_shape=result.padded_shape,
        runtime_ms=result.runtime_ms,
        device=result.device,
        threshold_used=result.threshold_used,
        class_histogram=result.class_histogram,
    )
    return JSONResponse(content=summary.dict(), media_type="application/json")
