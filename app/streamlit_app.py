import io
import tempfile
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import requests
import streamlit as st

from src.api.inference_service import SegmentationService
from src.api.settings import Settings


st.set_page_config(page_title="UNet3D - Demo de Segmentación", layout="wide")
st.title("UNet3D · Segmentación 3D")
st.caption("Sube un volumen NIfTI y genera la máscara con la API o de forma local.")


def _display_slice(mask: np.ndarray, axis: int, index: int, title: str) -> None:
    import matplotlib.pyplot as plt  # import lazy para no bloquear carga

    fig, ax = plt.subplots()
    if axis == 0:
        ax.imshow(mask[index, :, :], cmap="viridis")
    elif axis == 1:
        ax.imshow(mask[:, index, :], cmap="viridis")
    else:
        ax.imshow(mask[:, :, index], cmap="viridis")
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)


def _load_mask_from_bytes(raw: bytes) -> np.ndarray:
    bio = io.BytesIO(raw)
    img = nib.load(bio)
    return img.get_fdata().astype(np.uint8)


def infer_via_api(
    api_url: str, uploaded_file, threshold: Optional[float]
) -> Optional[np.ndarray]:
    if not uploaded_file:
        st.warning("Sube un archivo NIfTI para comenzar.")
        return None

    params = {"return_binary": True}
    if threshold is not None:
        params["threshold"] = threshold

    files = {"file": (uploaded_file.name, uploaded_file.getbuffer(), uploaded_file.type or "application/octet-stream")}
    try:
        with st.spinner("Llamando a la API..."):
            resp = requests.post(f"{api_url.rstrip('/')}/v1/predict", files=files, params=params, timeout=180)
        if not resp.ok:
            st.error(f"Error {resp.status_code}: {resp.text}")
            return None
        return _load_mask_from_bytes(resp.content)
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"No se pudo invocar la API: {exc}")
        return None


def infer_locally(
    uploaded_file, settings: Settings, threshold: Optional[float]
) -> Optional[np.ndarray]:
    if not uploaded_file:
        st.warning("Sube un archivo NIfTI para comenzar.")
        return None

    service = SegmentationService(settings=settings)
    try:
        with st.spinner("Cargando modelo..."):
            service.load_model()
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"No se pudo cargar el modelo: {exc}")
        return None

    suffix = ".nii.gz" if uploaded_file.name.lower().endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        with st.spinner("Ejecutando inferencia local..."):
            result = service.predict(tmp_path, threshold=threshold)
        return result.mask
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Fallo durante la inferencia: {exc}")
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


tab_api, tab_local = st.tabs(["Via API HTTP", "Local (sin red)"])

with tab_api:
    st.subheader("Usar API FastAPI")
    api_url = st.text_input("URL base de la API", value="http://localhost:8000")
    thr = st.slider("Umbral (solo binaria)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    api_file = st.file_uploader("Archivo NIfTI (.nii o .nii.gz)", type=["nii", "nii.gz"], key="api_uploader")
    if st.button("Inferir con API"):
        mask = infer_via_api(api_url=api_url, uploaded_file=api_file, threshold=thr)
        if mask is not None:
            st.success(f"Máscara recibida. Shape {mask.shape}")
            axis = st.selectbox("Eje para visualizar", options=[0, 1, 2], index=2)
            idx = st.slider("Corte", min_value=0, max_value=mask.shape[axis] - 1, value=mask.shape[axis] // 2)
            _display_slice(mask, axis=axis, index=idx, title=f"Corte {idx} (eje {axis})")
            uniq, counts = np.unique(mask, return_counts=True)
            st.write({int(k): int(v) for k, v in zip(uniq, counts)})

with tab_local:
    st.subheader("Inferencia local (misma máquina)")
    default_settings = Settings()
    weights_input = st.text_input("Ruta a checkpoint .pt/.pth", value=str(default_settings.model_path))
    thr_local = st.slider("Umbral (solo binaria)", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="thr_local")
    local_file = st.file_uploader(
        "Archivo NIfTI (.nii o .nii.gz)", type=["nii", "nii.gz"], key="local_uploader"
    )

    if st.button("Inferir en local"):
        local_settings = Settings(model_path=Path(weights_input))
        mask = infer_locally(uploaded_file=local_file, settings=local_settings, threshold=thr_local)
        if mask is not None:
            st.success(f"Máscara generada. Shape {mask.shape}")
            axis = st.selectbox("Eje para visualizar ", options=[0, 1, 2], index=2, key="local_axis")
            idx = st.slider(
                "Corte",
                min_value=0,
                max_value=mask.shape[axis] - 1,
                value=mask.shape[axis] // 2,
                key="local_idx",
            )
            _display_slice(mask, axis=axis, index=idx, title=f"Corte {idx} (eje {axis})")
            uniq, counts = np.unique(mask, return_counts=True)
            st.write({int(k): int(v) for k, v in zip(uniq, counts)})

st.markdown("---")
st.info(
    "1) Arranca la API con `uvicorn src.api.main:app --reload`.\n"
    "2) Carga tu checkpoint en `UNET3D_MODEL_PATH` o ajusta la ruta en la app.\n"
    "3) Sube un NIfTI y valida la salida con los cortes mostrados."
)
