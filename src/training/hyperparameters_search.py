import torch 
import numpy as np
from dataclasses import dataclass, asdict
import random
from typing import Dict, Optional, Any
import json
import time
import os

from src.model.diff_augment import *
from src.model.unet3d import *

def build_unet3d(hp):
    """
    Construye un modelo UNet3D a partir de un diccionario de hiperparámetros.

    Parámetros
    ----------
    hp : dict
        Debe incluir:
        - 'base' : int
        - 'norm' : {'in','bn',None}
        - 'dropout' : float
        Opcionales (con default):
        - 'in_channels' : int (default 1)
        - 'num_classes' : int (default 3)

    Retorna
    -------
    UNet3D
        Instancia configurada del modelo.
    """
    return UNet3D(in_channels = hp.get("in_channels", 1),
        num_classes = hp.get("num_classes", 3),
        base = hp["base"],
        norm  = hp["norm"],
        dropout  = hp["dropout"])


def build_optimizer(name: str, params, lr: float, weight_decay: float):
    """
    Crea un optimizador de PyTorch configurado por nombre.

    Parámetros
    ----------
    name : str
        'sgd' | 'adam' | 'adamw' (case-insensitive).
    params : iterable
        Parámetros del modelo (e.g., model.parameters()).
    lr : float
        Tasa de aprendizaje.
    weight_decay : float
        Decaimiento L2.

    Retorna
    -------
    torch.optim.Optimizer
        Instancia del optimizador solicitado.

    Excepciones
    -----------
    ValueError
        Si el optimizador no está soportado.
    """

    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    else:
        raise ValueError(f"Optimizer no soportado: {name}")

def build_augment(hp):
    """
    Construye un DiffAugment3D con flags y probabilidades desde un dict.

    Parámetros
    ----------
    hp : dict
        Debe incluir:
        - 'aug_p_flip' : float
        - 'aug_p_affine' : float
        - 'aug_brightness_contrast' : bool
        - 'aug_gamma' : bool
        - 'aug_noise' : bool

    Retorna
    -------
    DiffAugment3D
        Instancia de aumentaciones 3D para imagen+máscara.
    """

    return DiffAugment3D(
        p_flip  = hp["aug_p_flip"],
        p_affine= hp["aug_p_affine"],
        use_brightness_contrast = hp["aug_brightness_contrast"],
        use_gamma = hp["aug_gamma"],
        use_noise = hp["aug_noise"])


def _maybe_list_best(x):
    """
    Extrae el mejor valor (máximo) de una colección; si es escalar, lo convierte a float.

    Reglas
    ------
    - list/tuple: retorna max (ignorando NaN vía np.nanmax). Si vacío → -inf.
    - dict con clave 'values': usa max(dict['values']). Si vacío → -inf.
    - escalar: intenta float(x); si falla → -inf.

    Parámetros
    ----------
    x : Any
        Colección o escalar.

    Retorna
    -------
    float
        Máximo valor extraído o -inf si no aplicable.
    """

    if isinstance(x, (list, tuple)):
        return float(np.nanmax(np.array(x, dtype=float))) if len(x) else float("-inf")
    if isinstance(x, dict) and "values" in x:
        vals = x["values"]
        return float(np.nanmax(np.array(vals, dtype=float))) if len(vals) else float("-inf")
    try:
        return float(x)
    except Exception:
        return float("-inf")
    

@dataclass
class SearchSpaces:
    """
    Espacios de búsqueda para hiperparámetros.

    Atributos
    ---------
    bases : tuple[int]
        Canales base para la U-Net.
    norms : tuple[str]
        Tipos de normalización ('in', 'bn', 'gn'). *Ojo*: tu UNet3D actual solo implementa 'in'/'bn'/None.
    dropouts : tuple[float]
        Tasas de Dropout3d.
    optimizers : tuple[str]
        Nombres de optimizadores ('adamw', 'adam', 'sgd').
    lr_log10_min, lr_log10_max : float
        Rango log10 para muestrear la LR: lr = 10**U(min,max).
    wdecay_log10_min, wdecay_log10_max : float
        Rango log10 para muestrear weight decay: wd = 10**U(min,max).
    aug_p_flip : tuple[float, float]
        Rango para prob. de flips 3D.
    aug_p_affine : tuple[float, float]
        Rango para prob. de afín pequeña.
    aug_flags : tuple[tuple[bool,bool,bool], ...]
        Combinaciones (brightness_contrast, gamma, noise).
    """

    bases = (16, 32, 48, 64)
    norms = ("in", "bn", "gn")
    dropouts = (0.0, 0.1, 0.2, 0.3)

    optimizers = ("adamw", "adam", "sgd")
    lr_log10_min: float = -4.5
    lr_log10_max: float = -2.5
    wdecay_log10_min: float = -6
    wdecay_log10_max: float = -2

    aug_p_flip  = (0.2, 0.7)
    aug_p_affine= (0.0, 0.6)
    aug_flags = (
        (True, True, True),
        (True, False, True),
        (True, True, False),
        (False, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, False, False))


def sample_hparams(rng: np.random.Generator, spaces: SearchSpaces, in_channels=1, num_classes=3):
    """
    Muestrea un set de hiperparámetros desde los espacios definidos.

    Parámetros
    ----------
    rng : np.random.Generator
        Generador NumPy para muestreo reproducible.
    spaces : SearchSpaces
        Configuración de rangos y opciones.
    in_channels : int, opcional
        Canales de entrada del modelo. Por defecto 1.
    num_classes : int, opcional
        Número de clases de salida. Por defecto 3.

    Retorna
    -------
    dict
        Diccionario con hiperparámetros para modelo, optimizador y augmentaciones.
    """

    lr = 10 ** rng.uniform(spaces.lr_log10_min, spaces.lr_log10_max)
    wd = 10 ** rng.uniform(spaces.wdecay_log10_min, spaces.wdecay_log10_max)
    base = int(rng.choice(spaces.bases))
    norm = str(rng.choice(spaces.norms))
    dropout = float(rng.choice(spaces.dropouts))

    opt = str(rng.choice(spaces.optimizers))
    p_flip = float(rng.uniform(*spaces.aug_p_flip))
    p_aff  = float(rng.uniform(*spaces.aug_p_affine))
    b, g, n = random.choice(spaces.aug_flags)

    return dict(
        in_channels=in_channels, num_classes=num_classes,
        base=base, norm=norm, dropout=dropout,
        optimizer=opt, lr=lr, weight_decay=wd,
        aug_p_flip=p_flip, aug_p_affine=p_aff,
        aug_brightness_contrast=b, aug_gamma=g, aug_noise=n)


@dataclass
class TrialResult:
    """
    Resultado de un experimento/ensayo de búsqueda.

    Atributos
    ---------
    trial_id : int
        Identificador del trial.
    score : float
        Métrica objetivo (mayor es mejor).
    hp : Dict[str, Any]
        Hiperparámetros usados.
    train_time_s : float
        Tiempo de entrenamiento en segundos.
    ckpt_path : Optional[str]
        Ruta al checkpoint guardado (si aplica).
    """

    trial_id: int
    score: float
    hp: Dict[str, Any]
    train_time_s: float
    ckpt_path: Optional[str]


def set_all_seeds(seed:int):
    """
    Fija semillas para Python/NumPy/PyTorch (CPU y CUDA) para mayor reproducibilidad.

    Parámetros
    ----------
    seed : int
        Semilla base.

    Notas
    -----
    - No fuerza determinismo total (algunos kernels CUDA pueden seguir siendo no deterministas).
    - Si necesitas estricta reproducibilidad, considera `torch.use_deterministic_algorithms(True)`
      y variables de entorno relacionadas (puede impactar rendimiento).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


############################################################################

def hyperparam_search_unet3d(n_trials,
    device , criterion,trainer_fn,*,
    epocs: int = 20,
    patience: int = 8,
    min_delta: float = 1e-4,
    target_metric: float = 0.85,
    spaces: SearchSpaces = SearchSpaces(),
    seed: int = 2025, random_seed = True ,
    ckpt_dir: str = "ckpts_unet3d_search",
    run_name: str = "unet3d_search",
    save_every_trial: bool = True):

    """
    Random search de hiperparámetros para UNet3D con early stopping y guardado de checkpoints.

    Parámetros
    ----------
    n_trials : int
        Número de configuraciones a evaluar.
    device : str or torch.device
        Dispositivo para entrenamiento/inferencia.
    criterion : callable
        Función de pérdida (binaria o multiclase).
    trainer_fn : callable
        Función de entrenamiento que debe firmar:
        (model, optimizer, device, criterion, num_classes, epocs, augmnet, patience, min_delta, target_metric)
        y retornar (history_train: dict, history_val: dict).
    epocs : int, opcional
        Máximo de épocas por trial. Por defecto 20.
    patience : int, opcional
        Paciencia para early stopping. Por defecto 8.
    min_delta : float, opcional
        Mejora mínima para resetear paciencia. Por defecto 1e-4.
    target_metric : float, opcional
        Umbral de métrica para paro anticipado. Por defecto 0.85.
    spaces : SearchSpaces, opcional
        Espacios de muestreo para hiperparámetros. Por defecto SearchSpaces().
    seed : int, opcional
        Semilla base. Por defecto 2025.
    random_seed : bool, opcional
        Si True, sobreescribe `seed` con un entero aleatorio [1,1000]. Por defecto True.
    ckpt_dir : str, opcional
        Carpeta donde guardar checkpoints y resultados. Por defecto "ckpts_unet3d_search".
    run_name : str, opcional
        Prefijo para nombres de archivo. Por defecto "unet3d_search".
    save_every_trial : bool, opcional
        Si True, guarda checkpoint de cada trial. Por defecto True.

    Retorna
    -------
    tuple
        (best_hp: dict, best_score: float, results: list[TrialResult])
        Además:
        - Guarda mejor checkpoint en: {ckpt_dir}/{run_name}_BEST.pt
        - Guarda resumen JSON: {ckpt_dir}/{run_name}_results.json
    """

    if random_seed:
      seed = random.randint(1 , 1000)

    os.makedirs(ckpt_dir, exist_ok=True)
    set_all_seeds(seed)

    best_score = 0
    best_hp: Dict[str, Any] = {}
    best_path: Optional[str] = None
    results = []

    rng = np.random.default_rng(seed)

    for t in range(1, n_trials+1):

        # Random Choice de Hiperparametros
        hp = sample_hparams(rng, spaces)
        model = build_unet3d(hp).to(device)
        opt = build_optimizer(hp["optimizer"], model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        aug = build_augment(hp)

        # Entrenamiento
        t0 = time.time()

        hist_tr, hist_va = trainer_fn(
            model, opt, device, criterion,
            num_classes=hp["num_classes"],
            epocs=epocs,
            augmnet=aug,
            patience=patience,
            min_delta=min_delta,
            target_metric=target_metric)

        dt = time.time() - t0

        score =  max(hist_va[m]["mIoU"] for m in hist_va)

        ckpt_path = None
        if save_every_trial:
            ckpt_path = os.path.join(ckpt_dir, f"{run_name}_trial{t:03d}.pt")
            torch.save({"model_state": model.state_dict(), "hp": hp, "score": score}, ckpt_path)

        if score > best_score:
            best_score = score
            best_hp = hp
            best_path = os.path.join(ckpt_dir, f"{run_name}_BEST.pt")
            torch.save({"model_state": model.state_dict(), "hp": hp, "score": score}, best_path)

        results.append(TrialResult(trial_id=t, score=score, hp=hp, train_time_s=dt, ckpt_path=ckpt_path))

        print(f"[Trial {t:02d}/{n_trials}] score={score:.5f} | opt={hp['optimizer']} | lr={hp['lr']:.2e} | wd={hp['weight_decay']:.1e} | base={hp['base']} norm={hp['norm']} drop={hp['dropout']} | aug: flip={hp['aug_p_flip']:.2f}, aff={hp['aug_p_affine']:.2f}, bc={hp['aug_brightness_contrast']}, g={hp['aug_gamma']}, n={hp['aug_noise']} | {dt/60:.1f} min")

    with open(os.path.join(ckpt_dir, f"{run_name}_results.json"), "w") as f:
        json.dump(
            [{"trial_id": r.trial_id, "score": r.score, "train_time_s": r.train_time_s, "hp": r.hp, "ckpt_path": r.ckpt_path} for r in results],
            f, indent=2)

    print("\n[MEJOR MODELO]")
    print(json.dumps({"best_score": best_score, "best_hp": best_hp, "best_ckpt": best_path}, indent=2))
    return best_hp, best_score, results