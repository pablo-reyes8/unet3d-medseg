import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import os, nibabel as nib, numpy as np


def _stem_no_ext(p: Path):
    s = p.name
    if s.endswith(".nii.gz"):
        return s[:-7]
    return s[:s.rfind(".")] if "." in s else s

def minmax_normalize(img: np.ndarray, clip=None, eps=1e-6):
    """
    Normaliza una imagen 3D al rango [0, 1] usando escalado min–max opcionalmente recortado por percentiles.

    Parámetros
    ----------
    img : np.ndarray
        Imagen tridimensional (D, H, W) a normalizar.
    clip : tuple of float or None, opcional
        Percentiles para recorte antes de normalizar, por ejemplo (1, 99).
        Si es None, no se aplica recorte. Por defecto None.
    eps : float, opcional
        Pequeño valor para evitar divisiones por cero. Por defecto 1e-6.

    Retorna
    -------
    np.ndarray
        Imagen normalizada en el rango [0, 1] como float32.
        Si la imagen tiene varianza nula, retorna un arreglo de ceros.

    Ejemplo
    --------
    >>> minmax_normalize(img, clip=(1, 99)).min(), minmax_normalize(img, clip=(1, 99)).max()
    (0.0, 1.0)
    """
    x = img.astype(np.float32)
    if clip is not None:
        lo, hi = np.percentile(x, clip)
        if hi - lo < eps:
            lo, hi = x.min(), x.max()
        x = np.clip(x, lo, hi)

    minv = float(x.min())
    maxv = float(x.max())
    if maxv - minv < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - minv) / (maxv - minv)


### Data Loaders ####
class Hippocampus3DDataset(Dataset):
    """
    Dataset 3D para imágenes y máscaras del hipocampo en formato NIfTI.

    Permite cargar pares (imagen, máscara), aplicar normalización opcional
    y devolver tensores listos para entrenamiento en PyTorch.

    Parámetros
    ----------
    pairs : list of tuple(str, str)
        Lista de tuplas con las rutas (imagen, máscara) de los archivos NIfTI.
    norm : {'minmax', None}, opcional
        Método de normalización a aplicar. Por defecto 'minmax'.
        Si es None, no se normaliza.
    clip : tuple of float or None, opcional
        Percentiles usados para recorte robusto antes del min-max.
        Usa None para desactivar el recorte. Por defecto (1, 99).
    dtype_img : tipo de dato, opcional
        Tipo de dato usado al cargar la imagen. Por defecto np.float32.

    Métodos
    -------
    __len__():
        Retorna el número total de pares (imagen, máscara).
    __getitem__(idx):
        Carga el par correspondiente al índice `idx`, aplica normalización
        y retorna tensores (img, msk).

    Retorna
    -------
    tuple
        - img : torch.Tensor — Imagen 3D normalizada con shape (1, D, H, W).
        - msk : torch.Tensor — Máscara correspondiente, con tipo long.

    Excepciones
    -----------
    ValueError
        Si las dimensiones de la imagen y la máscara no coinciden,
        o si el método de normalización no está soportado.
    """

    def __init__(self, pairs, norm='minmax', clip=(1, 99), dtype_img=np.float32):
        """
        norm: 'minmax' (default) o None
        clip: percentiles para recorte robusto antes del min-max; usa None para desactivar
        """
        self.pairs = pairs
        self.norm = norm
        self.clip = clip
        self.dtype_img = dtype_img

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        img_nii = nib.load(str(img_path))
        msk_nii = nib.load(str(msk_path))
        img = nib.as_closest_canonical(img_nii).get_fdata().astype(self.dtype_img)
        msk = nib.as_closest_canonical(msk_nii).get_fdata().astype(np.int16)

        if img.shape != msk.shape:
            raise ValueError(f"Shape mismatch {img.shape} vs {msk.shape} en {img_path.name}")

        if self.norm == 'minmax':
            img = minmax_normalize(img, clip=self.clip)
        elif self.norm is None:
            img = img.astype(np.float32)
        else:
            raise ValueError(f"norm '{self.norm}' no soportada. Usa 'minmax' o None.")

        img = torch.from_numpy(img[None, ...]).float()
        msk = torch.from_numpy(msk).long()
        return img, msk