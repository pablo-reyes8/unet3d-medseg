import os, nibabel as nib, numpy as np, imageio.v2 as imageio
from pathlib import Path

def _minmax_uint8(x):
    """
    Escala un arreglo numérico al rango [0, 255] usando percentiles robustos y lo convierte a uint8.

    Parámetros
    ----------
    x : np.ndarray
        Arreglo numérico (por ejemplo, una imagen) de cualquier tipo numérico.

    Retorna
    -------
    np.ndarray
        Arreglo del mismo tamaño que `x`, normalizado y convertido a tipo uint8.

    Detalles
    --------
    - Usa los percentiles 1 y 99 para definir los límites de escala, reduciendo la
      influencia de valores atípicos.
    - Los valores fuera del rango se recortan (clip) entre 0 y 1 antes de escalar a 255.
    """
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    return (x * 255).astype(np.uint8)


def _blend_gray_mask(gray, mask, color=(255, 64, 64), alpha=0.35):
    """
    Superpone una máscara binaria sobre una imagen en escala de grises.

    Parámetros
    ----------
    gray : np.ndarray
        Imagen en escala de grises (H, W) tipo uint8.
    mask : np.ndarray
        Máscara binaria (H, W), con valores True o >0 donde se aplica el color.
    color : tuple of int, opcional
        Color RGB de la superposición (por defecto rojo claro: (255, 64, 64)).
    alpha : float, opcional
        Nivel de transparencia de la superposición (0 = transparente, 1 = opaco).

    Retorna
    -------
    np.ndarray
        Imagen RGB (H, W, 3) tipo uint8 con la máscara coloreada superpuesta.
    """
    rgb = np.stack([gray]*3, axis=-1).astype(np.float32)
    m = (mask > 0)[..., None].astype(np.float32)
    col = np.array(color, dtype=np.float32)
    rgb = (1 - alpha*m)*rgb + alpha*m*col
    return rgb.clip(0,255).astype(np.uint8)

def _iterate_slices(vol, axis=2):
    """
    Itera sobre cortes 2D de un volumen 3D a lo largo de un eje específico.

    Parámetros
    ----------
    vol : np.ndarray
        Volumen tridimensional (H, W, D).
    axis : int, opcional
        Eje a lo largo del cual se generan los cortes (0, 1 o 2). Por defecto 2.

    Genera
    -------
    np.ndarray
        Cortes bidimensionales (H, W) del volumen, uno por iteración.
    """
    if axis == 0:
        for i in range(vol.shape[0]): yield vol[i, :, :]
    elif axis == 1:
        for i in range(vol.shape[1]): yield vol[:, i, :]
    else:
        for i in range(vol.shape[2]): yield vol[:, :, i]

def make_gif_for_file(fname , IMG_DIR , LBL_DIR ,OUT_DIR , with_overlay=True, fps=12):
    """
    Genera GIFs animados de cortes 2D (axial, coronal y sagital) a partir de un volumen NIfTI.

    Parámetros
    ----------
    fname : str
        Nombre del archivo NIfTI (debe existir en IMG_DIR).
        Si hay una máscara con el mismo nombre en LBL_DIR, se usará como superposición.
    with_overlay : bool, opcional
        Si es True, superpone la máscara (si existe) sobre la imagen. Por defecto True.
    fps : int, opcional
        Cuadros por segundo del GIF. Por defecto 12.

    Detalles
    --------
    - Carga la imagen y su máscara (si existe) desde `IMG_DIR` y `LBL_DIR`.
    - Genera GIFs a lo largo de los tres ejes anatómicos:
      * Eje 2 → Axial (tag: "axialZ")
      * Eje 1 → Coronal (tag: "coronalY")
      * Eje 0 → Sagital (tag: "sagittalX")
    - Usa `_minmax_uint8` para escalar la intensidad y `_blend_gray_mask` para la superposición.

    Retorna
    -------
    None
        Guarda los GIFs generados en `OUT_DIR` con nombres descriptivos.

    Ejemplo
    --------
    >>> make_gif_for_file("subject001.nii.gz")
    Guardado: OUT_DIR/subject001_axialZ.gif
    Guardado: OUT_DIR/subject001_coronalY.gif
    Guardado: OUT_DIR/subject001_sagittalX.gif
    """
    img_path = os.path.join(IMG_DIR, fname)
    img = np.asanyarray(nib.load(img_path).dataobj)

    # mascara
    msk_path = os.path.join(LBL_DIR, fname)
    mask = None
    if os.path.exists(msk_path):
        mask = np.asanyarray(nib.load(msk_path).dataobj)

    for axis, tag in zip([2,1,0], ["axialZ","coronalY","sagittalX"]):
        frames = []
        for sl in _iterate_slices(img, axis=axis):
            g = _minmax_uint8(sl)
            if with_overlay and (mask is not None):
                # slice correspondiente en la máscara
                if axis == 0: ms = mask[frames.__len__(), :, :]
                elif axis == 1: ms = mask[:, frames.__len__(), :]
                else: ms = mask[:, :, frames.__len__()]
                frame = _blend_gray_mask(g, ms)
            else:
                frame = np.stack([g]*3, axis=-1)
            frames.append(frame)
        out = OUT_DIR / f"{Path(fname).stem}_{tag}.gif"
        imageio.mimsave(out, frames, duration=1.0/max(fps,1))
        print(f"Guardado: {out}")


import glob ,base64
from IPython.display import Image, display
from IPython.display import HTML

def embed_gif(path, width=500):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"<img src='data:image/gif;base64,{b64}' style='width:{width}px;height:auto;border-radius:8px;'>"