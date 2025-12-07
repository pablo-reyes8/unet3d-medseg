import os, re, nibabel as nib
import numpy as np
from collections import defaultdict

def _valid_nii_list(path):
    """
    Obtiene y ordena los archivos NIfTI válidos (.nii o .nii.gz) dentro de un directorio.

    Parámetros
    ----------
    path : str
        Ruta del directorio donde se buscan los archivos NIfTI.

    Retorna
    -------
    list of str
        Lista de nombres de archivos NIfTI válidos, ordenados numéricamente
        según el primer número encontrado en el nombre del archivo.

    Detalles
    --------
    - Ignora archivos temporales que comienzan con '._'.
    - Si el nombre no contiene números, se usa el nombre completo como clave de ordenamiento.
    """
    files = [
        f for f in os.listdir(path)
        if (f.endswith((".nii", ".nii.gz")) and not f.startswith("._"))]
    def key_fn(fn):
        m = re.search(r'(\d+)', fn)
        return int(m.group(1)) if m else fn
    return sorted(files, key=key_fn)


def basename_noext(fn):
    return re.sub(r'\.nii(\.gz)?$', '', fn)

def quick_meta(nifti_path):
    """
    Extrae metadatos básicos de un archivo NIfTI (.nii o .nii.gz).

    Parámetros
    ----------
    nifti_path : str
        Ruta al archivo NIfTI.

    Retorna
    -------
    tuple
        Una tupla con tres elementos:
        - shape : tuple — Dimensiones del volumen (x, y, z, ...).
        - zooms : tuple — Tamaño del voxel en cada eje.
        - dtype : numpy.dtype — Tipo de dato de la imagen.
    """
    img = nib.load(nifti_path)
    shape = img.shape
    zooms = img.header.get_zooms()
    dtype = img.get_data_dtype()
    return shape, zooms, dtype


def list_valid(path):
    return sorted([f for f in os.listdir(path)
                   if (f.endswith((".nii",".nii.gz")) and not f.startswith("._"))])

def bname(fn):
    return re.sub(r'\.nii(\.gz)?$', '', fn)

def quick_subsample_stats(nifti_path, step=4):
    """
    Calcula estadísticas básicas de un archivo NIfTI usando submuestreo para un chequeo rápido.

    Parámetros
    ----------
    nifti_path : str
        Ruta al archivo NIfTI (.nii o .nii.gz).
    step : int, opcional
        Intervalo de submuestreo entre voxeles (por defecto 4).

    Retorna
    -------
    dict
        Diccionario con las siguientes claves:
        - 'empty' : bool — Indica si el volumen está vacío.
        - 'finite_ok' : bool — True si todos los valores submuestreados son finitos.
        - 'var' : float — Varianza del volumen submuestreado (0.0 si no hay valores finitos).

    Notas
    -----
    El submuestreo permite evaluar la calidad del volumen sin cargarlo completamente en memoria.
    """

    img = nib.load(nifti_path)
    arr = np.asanyarray(img.dataobj)[::step, ::step, ::step]
    if arr.size == 0:
        return {"empty": True, "finite_ok": False, "var": 0.0}
    finite = np.isfinite(arr)
    var = float(arr.var()) if finite.any() else 0.0
    return {"empty": False, "finite_ok": finite.all(), "var": var}



def check_files(dir_path, files, label_mode=False, max_list=5):
    """
    Verifica la integridad y validez básica de una lista de archivos NIfTI en un directorio.

    Parámetros
    ----------
    dir_path : str
        Ruta del directorio donde se encuentran los archivos.
    files : list of str
        Lista de nombres de archivos NIfTI a verificar.
    label_mode : bool, opcional
        Si es True, omite el chequeo de varianza (útil para máscaras o etiquetas).
        Por defecto es False.
    max_list : int, opcional
        Número máximo de ejemplos a mostrar en el resumen de errores. Por defecto 5.

    Retorna
    -------
    tuple
        - bad : dict — Diccionario con listas de archivos problemáticos clasificados por tipo:
          - 'size0': archivos vacíos.
          - 'corrupt': archivos que no pudieron cargarse.
          - 'naninf': contienen valores NaN o infinitos.
          - 'zerovar': con varianza prácticamente nula (solo si `label_mode=False`).
          - 'emptyslice': con dimensiones vacías o sin voxeles válidos.
        - summary : str — Resumen legible con conteos y ejemplos de cada categoría.

    Notas
    -----
    Esta función usa `quick_subsample_stats` para realizar comprobaciones rápidas sin cargar
    completamente los volúmenes en memoria.
    """
    bad = {"size0":[], "corrupt":[], "naninf":[], "zerovar":[], "emptyslice":[]}
    for f in files:
        fp = os.path.join(dir_path, f)

        if os.path.getsize(fp) == 0:
            bad["size0"].append(f);continue

        try:
            img = nib.load(fp)
            shp = img.shape
            if any(s == 0 for s in shp):
                bad["emptyslice"].append((f, shp))

            st = quick_subsample_stats(fp, step=4)
            if st["empty"]:
                bad["emptyslice"].append((f, shp))

            if not st["finite_ok"]:
                bad["naninf"].append(f)

            if not label_mode and st["var"] < 1e-8:
                bad["zerovar"].append(f)
        except Exception as e:
            bad["corrupt"].append(f)

    def show(tag):
        arr = bad[tag]
        return f"{tag}: {len(arr)}" + (f" (ej: {arr[:max_list]})" if arr else "")
    return bad, " | ".join([show(k) for k in ["size0","corrupt","naninf","zerovar","emptyslice"]])


def valid_list(p):
    return sorted([f for f in os.listdir(p) if f.endswith((".nii",".nii.gz")) and not f.startswith("._")])

def bname(fn):
    return re.sub(r'\.nii(\.gz)?$', '', fn)


def pad_center(arr, target):
    """
    Rellena (padding) un arreglo Numpy con ceros para centrarlo dentro de un tamaño objetivo.

    Parámetros
    ----------
    arr : np.ndarray
        Arreglo original a rellenar.
    target : tuple of int
        Forma (shape) objetivo a la que se desea ajustar el arreglo.

    Retorna
    -------
    np.ndarray
        Arreglo con padding aplicado, centrado dentro del tamaño objetivo.
        Si una dimensión ya es mayor o igual que la deseada, no se recorta ni modifica.

    Ejemplo
    --------
    >>> arr.shape
    (60, 60, 60)
    >>> pad_center(arr, (64, 64, 64)).shape
    (64, 64, 64)
    """
    pad = []
    for s, t in zip(arr.shape, target):
        total = max(t - s, 0)
        left = total // 2
        right = total - left
        pad.append((left, right))
    return np.pad(arr, pad_width=pad, mode="constant", constant_values=0)


def bbox_from_mask(mask):
    """
    Calcula el tamaño del recuadro delimitador (bounding box) de una máscara 3D binaria.

    Parámetros
    ----------
    mask : np.ndarray
        Máscara tridimensional (H, W, D) con valores >0 donde existe señal.

    Retorna
    -------
    tuple or None
        Tupla (dx, dy, dz) con el tamaño del bounding box en cada eje.
        Retorna None si la máscara está vacía (sin valores >0).

    Ejemplo
    --------
    >>> bbox_from_mask(mask)
    (45, 38, 22)
    """
    pos = np.where(mask > 0)
    if len(pos[0]) == 0: return None
    x_min, x_max = pos[0].min(), pos[0].max()
    y_min, y_max = pos[1].min(), pos[1].max()
    z_min, z_max = pos[2].min(), pos[2].max()
    return (x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1)