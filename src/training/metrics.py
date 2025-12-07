
import torch


def spatial_dims(x: torch.Tensor):
    """
    Devuelve las dimensiones espaciales de un tensor (excluye el batch).

    Parámetros
    ----------
    x : torch.Tensor
        Tensor de entrada.

    Retorna
    -------
    tuple
        Tupla con los ejes 1..(ndim-1). Si x.ndim <= 2, retorna ().
    """
    if x.ndim <= 2:
        return tuple()

    return tuple(range(1, x.ndim))

def ensure_binary_target_3d(y: torch.Tensor):
    """
    Asegura objetivo binario con canal explícito y tipo float.

    - Si y tiene forma [B, D, H, W], añade canal → [B, 1, D, H, W].
    - Convierte a float para métricas continuas.

    Parámetros
    ----------
    y : torch.Tensor
        Máscara binaria 3D por batch.

    Retorna
    -------
    torch.Tensor
        Tensor con canal explícito y dtype float.
    """
    if y.ndim == 4:
        y = y.unsqueeze(1)
    return y.float()

def dice_coeff_3d(pred: torch.Tensor, target: torch.Tensor, eps=1e-7):
    """
    Dice binario promedio en 3D por batch.

    Parámetros
    ----------
    pred : torch.Tensor
        Predicción binaria en {0,1}, forma [B, 1, D, H, W].
    target : torch.Tensor
        Objetivo binario en {0,1}, forma [B, 1, D, H, W].
    eps : float, opcional
        Estabilizador numérico. Por defecto 1e-7.

    Retorna
    -------
    torch.Tensor
        Escalar con el Dice medio en el batch.
    """
    reduce_dims = spatial_dims(pred)
    inter = (pred * target).sum(dim=reduce_dims)
    union = pred.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()

def mean_iou_mc_3d(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps=1e-7):
    """
    mIoU multiclase 3D, promediando solo sobre clases presentes (union > 0).

    Parámetros
    ----------
    pred : torch.Tensor
        Predicción entera {0..C-1}, forma [B, D, H, W].
    target : torch.Tensor
        Objetivo entero {0..C-1}, forma [B, D, H, W].
    num_classes : int
        Número total de clases C.
    eps : float, opcional
        Estabilizador numérico. Por defecto 1e-7.

    Retorna
    -------
    float
        mIoU promedio sobre clases con unión positiva.
    """
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        targ_c = (target == c)
        inter = (pred_c & targ_c).sum().float()
        union = (pred_c | targ_c).sum().float()
        if union > 0:
            ious.append((inter + eps) / (union + eps))
    return torch.stack(ious).mean().item() if ious else 0.0