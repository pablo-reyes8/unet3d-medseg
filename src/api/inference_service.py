import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import torch

from src.api.settings import Settings
from src.data.dataloaders import minmax_normalize
from src.model.unet3d import UNet3D


@dataclass
class PredictionResult:
    mask: np.ndarray
    mask_bytes: bytes
    affine: np.ndarray
    input_shape: Tuple[int, int, int]
    padded_shape: Tuple[int, int, int]
    runtime_ms: float
    device: str
    filename: str
    threshold_used: float
    class_histogram: Dict[int, int]


class SegmentationService:
    """
    Servicio de inferencia para UNet3D que centraliza la carga del modelo y el
    preprocesamiento de volúmenes NIfTI.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device: torch.device = torch.device("cpu")
        self._model: Optional[torch.nn.Module] = None

    @property
    def model_ready(self) -> bool:
        return self._model is not None

    def load_model(self, force_reload: bool = False) -> torch.nn.Module:
        """Carga el checkpoint en memoria si aún no está disponible."""
        if self._model is not None and not force_reload:
            return self._model

        device_str = self._pick_device()
        self.device = torch.device(device_str)

        ckpt_path = self.settings.model_path
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No se encontró el checkpoint en {ckpt_path}. "
                "Configura UNET3D_MODEL_PATH o actualiza settings.model_path."
            )

        state = torch.load(str(ckpt_path), map_location=self.device)
        state_dict = state.get("model", state) if isinstance(state, dict) else state
        if all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        model = UNet3D(
            in_channels=self.settings.in_channels,
            num_classes=self.settings.num_classes,
            base=self.settings.base_channels,
            norm=self.settings.norm,
            dropout=self.settings.dropout,
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    def predict(self, volume_path: Path, threshold: Optional[float] = None) -> PredictionResult:
        """
        Ejecuta inferencia sobre un archivo NIfTI y devuelve la máscara predicha y metadatos.
        """
        model = self.load_model()
        threshold = float(threshold) if threshold is not None else float(self.settings.default_threshold)

        volume, affine = self._load_volume(volume_path)
        input_shape = volume.shape

        normed = minmax_normalize(volume, clip=self.settings.clip_percentiles)
        padded, pad_info = self._pad_to_multiple(normed)
        padded_shape = padded.shape

        x = torch.from_numpy(padded[None, None, ...]).float().to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            logits = model(x)
            if self.settings.num_classes == 1:
                probs = torch.sigmoid(logits)
                pred = (probs > threshold).to(torch.uint8)
            else:
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1, keepdim=True).to(torch.uint8)
        runtime_ms = (time.perf_counter() - start) * 1000.0

        pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
        pred_np = self._remove_padding(pred_np, pad_info)

        hist = self._class_histogram(pred_np)

        nifti_img = nib.Nifti1Image(pred_np.astype(np.uint8), affine=affine)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            nib.save(nifti_img, str(tmp_path))
            mask_bytes = tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

        filename = f"{self._basename(volume_path)}_mask.nii.gz"
        return PredictionResult(
            mask=pred_np,
            mask_bytes=mask_bytes,
            affine=affine,
            input_shape=input_shape,
            padded_shape=padded_shape,
            runtime_ms=runtime_ms,
            device=str(self.device),
            filename=filename,
            threshold_used=threshold,
            class_histogram=hist,
        )

    def _load_volume(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo {path}")
        img = nib.load(str(path))
        canonical = nib.as_closest_canonical(img)
        data = canonical.get_fdata().astype(np.float32)
        if data.ndim == 4 and data.shape[-1] == 1:
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f"Se esperaba un volumen 3D; shape recibido {data.shape}")
        return data, canonical.affine

    def _pick_device(self) -> str:
        if self.settings.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.settings.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.settings.device

    def _pad_to_multiple(self, arr: np.ndarray) -> Tuple[np.ndarray, Tuple[Tuple[int, int], ...]]:
        factor = max(1, int(self.settings.pad_multiple))
        pad_cfg = []
        for dim in arr.shape:
            remainder = dim % factor
            if remainder == 0:
                pad_cfg.append((0, 0))
            else:
                total = factor - remainder
                left = total // 2
                right = total - left
                pad_cfg.append((left, right))
        padded = np.pad(arr, pad_cfg, mode="constant", constant_values=0)
        return padded, tuple(pad_cfg)

    def _remove_padding(self, arr: np.ndarray, pad_cfg: Tuple[Tuple[int, int], ...]) -> np.ndarray:
        slices = []
        for dim_pad in pad_cfg:
            left, right = dim_pad
            start = left
            end = arr.shape[len(slices)] - right if right > 0 else None
            slices.append(slice(start, end))
        return arr[tuple(slices)]

    def _class_histogram(self, arr: np.ndarray) -> Dict[int, int]:
        unique, counts = np.unique(arr, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def _basename(self, path: Path) -> str:
        name = path.name
        if name.endswith(".nii.gz"):
            return name[:-7]
        if name.endswith(".nii"):
            return name[:-4]
        return path.stem
