# UNet3D Segmentation Suite

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
- Global IoU: ![IoU](experiments/IoU.png)
- IoU across depth: ![IoU Z-axis](experiments/IoU%20image%20z%20axis.png)
- Sample predictions: ![Predictions](experiments/model%20predictions.png)
- Error overlays: ![Overlay](experiments/overlay%20errors.png)

## License
MIT License. You are free to use, modify, and distribute with attribution and without warranty.

## Support
For issues or improvements, open a ticket or reach the maintainers listed in `model.yaml`.
