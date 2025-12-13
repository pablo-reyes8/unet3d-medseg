# UNet3D Segmentation Suite

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/unet3d-medseg)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/unet3d-medseg)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/unet3d-medseg)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/unet3d-medseg)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/unet3d-medseg?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/unet3d-medseg?style=social)

End-to-end stack for 3D medical image segmentation with a UNet3D backbone. The project is dual-purpose: it contains the training code for volumetric segmentation and a production-grade deployment path (FastAPI + Docker) with an interactive Streamlit front-end for human review.

## Highlights
- 3D UNet architecture with min–max normalization, percentile clipping, and padding to keep volumes aligned with encoder strides.
- FastAPI service in `src/api` for file-based inference (NIfTI in, NIfTI out) with clean OpenAPI docs and CORS.
- Streamlit app in `app/` to upload volumes, call the API or run local inference, and visualize slices with class histograms.
- Container-ready via `Dockerfile`; configurable through environment variables (see `UNET3D_*` settings).
- Model metadata captured in `model.yaml` for quick discoverability and auditing.

## Repository Structure
```
├─ app/                     # Streamlit front-end for upload + visualization
├─ experiments/             # Evaluation figures and qualitative samples
├─ notebooks/               # Exploratory notebooks
├─ src/
│  ├─ api/                  # FastAPI app, schemas, settings, inference service
│  ├─ data/                 # NIfTI loaders, preprocessing helpers
│  ├─ model/                # UNet3D and building blocks
│  ├─ model_inference.py/   # Analysis utilities (qualitative and posterior analysis)
│  └─ training/             # Training loop, metrics, hyperparameter search
├─ model.yaml               # Model card with metadata, I/O contract, serving info
├─ requirements.txt         # Python dependencies
└─ Dockerfile               # Container for the FastAPI service
```

## Model Card (summary)
- Name: `unet3d-segmentation` (`model.yaml`)
- Task: 3D medical segmentation (NIfTI volumes)
- Inputs: single-channel volume; min–max normalization with percentiles (1,99); padded to multiples of 16
- Outputs: NIfTI mask with 3 classes (background + 2 foreground classes by default)
- Checkpoint: `artifacts/unet3d_best.pt` (configurable via `UNET3D_MODEL_PATH`)
- Serving: `src.api.main:app` on port `8000`; main endpoint `/v1/predict`
- License: MIT (see README)

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate  # or activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=.
export UNET3D_MODEL_PATH=/absolute/path/to/your_checkpoint.pt  # required for inference
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### API usage
- Docs: `http://localhost:8000/docs` or `http://localhost:8000/redoc`
- Health: `GET /health`
- Predict (returns JSON by default):
```bash
curl -X POST "http://localhost:8000/v1/predict" \
  -H "accept: application/json" \
  -F "file=@/path/to/volume.nii.gz"
```
- Predict and download mask:
```bash
curl -X POST "http://localhost:8000/v1/predict?return_binary=true" \
  -H "accept: application/octet-stream" \
  -F "file=@/path/to/volume.nii.gz" \
  -o mask.nii.gz
```
- Optional query `threshold` for binary models (default defined in settings).

### Streamlit front-end
```bash
export PYTHONPATH=.
streamlit run app/streamlit_app.py
```
Use the “Via API HTTP” tab to point to the FastAPI base URL, or “Local” to run inference on the same machine with the configured checkpoint.

## Docker Deployment
Build and run the API container:
```bash
docker build -t unet3d-api .
docker run --rm -p 8000:8000 \
  -e UNET3D_MODEL_PATH=/models/unet3d_best.pt \
  -v /absolute/path/to/checkpoints:/models \
  unet3d-api
```
Environment variables supported (all prefixed with `UNET3D_`):
- `MODEL_PATH`: checkpoint path inside the container (default `/models/unet3d_best.pt`)
- `DEVICE`: `auto` | `cuda` | `cpu` (defaults to auto-select)
- `DEFAULT_THRESHOLD`, `PAD_MULTIPLE`, `CLIP_PERCENTILES`, `ALLOW_ORIGINS`, etc. (see `src/api/settings.py`)

## Training Notes
- Training utilities live in `src/training/` and expect paired NIfTI volumes and masks with matching shapes.
- Normalization uses min–max with optional percentile clipping; pad volumes to keep dimensions divisible by 16 for the encoder/decoder strides.
- Example starting point: adapt `train_unet.py` to your dataloaders and call `train_uneted(...)` with early stopping/patience parameters from `model.yaml`.

## Visual Results


<h2 align="center">Visual Results</h2>

<table align="center">
  <!-- Top row: smaller, context metrics -->
  <tr>
    <td align="center" width="50%">
      <b>IoU by class (validation)</b><br/>
      <img src="experiments/IoU.png" width="360" alt="IoU by class (validation)">
    </td>
    <td align="center" width="50%">
      <b>IoU across volume (validation)</b><br/>
      <img src="experiments/IoU%20image%20z%20axis.png" width="360" alt="IoU across volume (validation)">
    </td>
  </tr>

  <!-- Second row: larger, main qualitative results -->
  <tr>
    <td align="center" colspan="2">
      <b>Sample predictions</b><br/>
      <img src="experiments/model%20predictions.png" width="820" alt="Sample predictions">
    </td>
  </tr>

  <tr>
    <td align="center" colspan="2">
      <b>Error overlays</b><br/>
      <img src="experiments/overlay%20errors.png" width="720" alt="Error overlays">
    </td>
  </tr>
</table>

## License
MIT License. You are free to use, modify, and distribute with attribution and without warranty.

## Support
For issues or improvements, open a ticket or reach the maintainers listed in `model.yaml`.
