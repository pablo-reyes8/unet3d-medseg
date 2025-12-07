import os, nibabel as nib, numpy as np
import plotly.graph_objects as go

ROOT = "DATA ROOT"
IMG_DIR_CAND = [os.path.join(ROOT, "imagesTr_norm"), os.path.join(ROOT, "imagesTr_pad")]
LBL_DIR = os.path.join(ROOT, "labelsTr_pad")

for d in IMG_DIR_CAND:
    if os.path.isdir(d) and any(f.endswith((".nii",".nii.gz")) for f in os.listdir(d)):
        IMG_DIR = d
        break

fname = sorted([f for f in os.listdir(IMG_DIR) if f.endswith((".nii",".nii.gz"))])[0]
img = np.asanyarray(nib.load(os.path.join(IMG_DIR, fname)).dataobj).astype(np.float32)
msk = np.asanyarray(nib.load(os.path.join(LBL_DIR, fname)).dataobj)

STEP = 2
img_ds = img[::STEP, ::STEP, ::STEP]
msk_ds = msk[::STEP, ::STEP, ::STEP]

vals = img_ds[img_ds > 0]
p1, p99 = (np.percentile(vals, 1), np.percentile(vals, 99)) if vals.size else (img_ds.min(), img_ds.max())
img_vis = np.clip((img_ds - p1) / max(p99 - p1, 1e-6), 0, 1)

nx, ny, nz = img_vis.shape
x = np.arange(nx); y = np.arange(ny); z = np.arange(nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

fig = go.Figure()

fig.add_trace(go.Volume(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    value=img_vis.flatten(),
    opacity=0.08,
    surface_count=12,
    colorscale='Gray',
    showscale=False,))

for cls, cmap, op in [(1, 'Reds', 0.45), (2, 'Blues', 0.45)]:
    binm = (msk_ds == cls).astype(np.float32)
    if binm.sum() == 0:
        continue

    fig.add_trace(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=binm.flatten(),
        isomin=0.5, isomax=1.0,
        surface_count=1,
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=op,
        colorscale=cmap,
        showscale=False,
        name=f"mask_{cls}",))

fig.update_layout(
    title=f"3D Volume Rendering — {fname}",
    scene=dict(
        xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
        aspectmode="data"),
    margin=dict(l=0, r=0, t=40, b=0),)

fig.show()