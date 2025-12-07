import torch
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def error_overlay_3d(
    model, dataset, device, num_classes=3,
    idx=None, z=None, class_id=None,
    alpha=0.8):
    """
    Visualiza FP/FN sobre un corte Z de un volumen 3D.

    Parámetros
    ----------
    model : nn.Module
        Modelo de segmentación 3D (logits B×C×D×H×W).
    dataset : torch.utils.data.Dataset
        Devuelve (img, gt) con img (1, D, H, W) y gt (D, H, W).
    device : str or torch.device
        Dispositivo de inferencia.
    num_classes : int, opcional
        Total de clases (informativo). Por defecto 3.
    idx : int or None, opcional
        Índice del volumen a mostrar. Aleatorio si None.
    z : int or None, opcional
        Índice del corte Z. Mitad del volumen si None.
    class_id : int or None, opcional
        Si None, evalúa “cualquier clase ≠ fondo” (fondo=0). Si se especifica,
        calcula FP/FN sólo para esa clase.
    alpha : float, opcional
        Transparencia del overlay. Por defecto 0.8.

    Mapa de errores
    ---------------
    - FP (azul): predijo clase (o `class_id`) donde GT es otra.
    - FN (rojo): GT es clase (o `class_id`) y la predicción dijo otra.

    Efecto
    ------
    Muestra dos paneles:
      (1) Overlay de FP/FN sobre la imagen (colormaps fríos/templados).
      (2) Overlay con paletas 'Blues'/'Reds' y conteos FP/FN.
    """

    model.eval()
    if idx is None:
        idx = np.random.randint(0, len(dataset))
    img, gt = dataset[idx]
    img = img.unsqueeze(0).to(device)
    gt_np = gt.numpy()

    # predicción
    logits = model(img)
    pred = logits.argmax(1).squeeze().cpu().numpy()
    img_np = img.squeeze().cpu().numpy()
    if z is None:
        z = img_np.shape[0] // 2

    # máscaras de error
    if class_id is None:
        fg_pred = pred != 0
        fg_gt   = gt_np != 0
        FP = np.logical_and(fg_pred, ~fg_gt)
        FN = np.logical_and(~fg_pred, fg_gt)
    else:
        P = pred == class_id
        G = gt_np == class_id
        FP = np.logical_and(P, ~G)
        FN = np.logical_and(~P, G)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    base = img_np[z]
    for j in range(2):
        ax[j].imshow(base, cmap='gray')
        ax[j].axis('off')

    ax[0].imshow(np.ma.masked_where(~FP[z], FP[z]), cmap='cool', alpha=alpha)
    ax[0].imshow(np.ma.masked_where(~FN[z], FN[z]), cmap='autumn', alpha=alpha)
    ax[0].set_title(f"Overlay errores (Z={z})")

    FPn = int(FP[z].sum()); FNn = int(FN[z].sum())
    ax[1].imshow(base, cmap='gray')
    ax[1].imshow(np.ma.masked_where(~FP[z], FP[z]), cmap='Blues', alpha=alpha)
    ax[1].imshow(np.ma.masked_where(~FN[z], FN[z]), cmap='Reds',  alpha=alpha)
    lbl = "Todas≠fondo" if class_id is None else f"Clase {class_id}"
    ax[1].set_title(f"{lbl} | FP={FPn:,} · FN={FNn:,}")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()