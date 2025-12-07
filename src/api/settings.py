from pathlib import Path
from typing import List, Tuple

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Configuración central de la API y del servicio de inferencia."""

    model_path: Path = Field(
        default=Path("artifacts/unet3d_best.pt"),
        description="Ruta al checkpoint .pt/.pth con los pesos entrenados.",
    )
    device: str = Field(
        default="auto",
        description="Dispositivo preferido ('auto', 'cuda', 'cpu').",
    )
    num_classes: int = Field(default=3, description="Número de clases de salida.")
    in_channels: int = Field(default=1, description="Canales de entrada del modelo.")
    base_channels: int = Field(default=32, description="Canales base en el primer bloque de la U-Net.")
    norm: str = Field(default="in", description="Tipo de normalización usada al entrenar ('in'|'bn'|None).")
    dropout: float = Field(default=0.0, description="Tasa de dropout usada al entrenar el modelo.")

    pad_multiple: int = Field(
        default=16,
        description="Alinea D/H/W al múltiplo indicado para evitar desajustes por downsampling.",
    )
    clip_percentiles: Tuple[int, int] = Field(
        default=(1, 99),
        description="Percentiles para recorte robusto previo a la normalización min-max.",
    )
    default_threshold: float = Field(
        default=0.5,
        description="Umbral por defecto para segmentación binaria (sigmoid > threshold).",
    )
    class_names: List[str] = Field(
        default_factory=lambda: ["background", "class-1", "class-2"],
        description="Etiquetas legibles de cada clase (índice = canal de salida).",
    )

    server_host: str = Field(default="0.0.0.0", description="Host para uvicorn.")
    server_port: int = Field(default=8000, description="Puerto para uvicorn.")
    docs_url: str = Field(default="/docs", description="Ruta del swagger UI.")
    redoc_url: str = Field(default="/redoc", description="Ruta de ReDoc UI.")
    openapi_url: str = Field(default="/openapi.json", description="Ruta del esquema OpenAPI.")

    allow_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS: orígenes permitidos para el front.",
    )

    class Config:
        env_prefix = "UNET3D_"
        env_file = ".env"
        case_sensitive = False

    @validator("device")
    def _device_validator(cls, value: str) -> str:
        allowed = {"auto", "cuda", "cpu"}
        if value.lower() not in allowed:
            raise ValueError(f"device debe ser uno de {allowed}")
        return value.lower()

