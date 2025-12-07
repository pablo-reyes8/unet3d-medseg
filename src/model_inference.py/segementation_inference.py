import torch 
import numpy as np
import matplotlib.pyplot as plt

def show_random_slice(model, dataset, device, num_classes=3, idx=None):
    """
    Muestra una predicción 2D (corte medio en Z) de un volumen del dataset.

    Parámetros
    ----------
    model : nn.Module
        Modelo de segmentación 3D.
    dataset : torch.utils.data.Dataset
        Devuelve pares (img, mask) con img de forma (1, D, H, W).
    device : str or torch.device
        Dispositivo de inferencia.
    num_classes : int, opcional
        1 para binaria (usa sigmoid+umbral 0.5), >1 para multiclase (argmax). Por defecto 3.
    idx : int or None, opcional
        Índice de la muestra; aleatorio si None.

    Efecto
    ------
    Renderiza tres paneles: imagen, máscara real y predicción (corte Z medio).
    """
    model.eval()
    if idx is None:
        idx = np.random.randint(0, len(dataset))
    img, mask = dataset[idx]
    img = img.unsqueeze(0).to(device)
    mask_np = mask.cpu().numpy()

    with torch.no_grad():
        logits = model(img)
        if num_classes == 1:
            pred = (torch.sigmoid(logits) > 0.5).float()
            pred_np = pred.squeeze().cpu().numpy()
        else:
            pred = logits.argmax(dim=1).squeeze()
            pred_np = pred.cpu().numpy()

    img_np = img.squeeze().cpu().numpy()

    mid_z = img_np.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np[mid_z], cmap='gray')
    axes[0].set_title("Imagen original")
    axes[1].imshow(mask_np[mid_z], cmap='viridis')
    axes[1].set_title("Máscara real")
    axes[2].imshow(pred_np[mid_z], cmap='viridis')
    axes[2].set_title("Predicción modelo")
    for a in axes:
      a.axis('off')
    plt.suptitle(f"Volumen #{idx}  |  Corte Z={mid_z}")
    plt.tight_layout()
    plt.show()

def compare_slices(mask_np, pred_np, img_np=None, slices=None, title_prefix=""):
    """
    Compara GT vs. predicción en varios cortes Z.

    Parámetros
    ----------
    mask_np : np.ndarray
        Máscara real con forma (D, H, W).
    pred_np : np.ndarray
        Predicción con forma (D, H, W).
    img_np : np.ndarray or None, opcional
        Imagen base (D, H, W) para el panel superior. Por defecto None.
    slices : list[int] or None, opcional
        Índices de cortes Z a mostrar; por defecto usa [D/4, D/2, 3D/4].
    title_prefix : str, opcional
        Título general de la figura.

    Efecto
    ------
    Dibuja rejilla 3×N: (imagen opcional, máscara, predicción).
    """

    if slices is None:
        D = mask_np.shape[0]
        slices = [D//4, D//2, 3*D//4]

    n = len(slices)
    fig, axes = plt.subplots(3, n, figsize=(4*n, 10))
    for i, z in enumerate(slices):
        if img_np is not None:
            axes[0, i].imshow(img_np[z], cmap='gray')
            axes[0, i].set_title(f"Imagen Z={z}")
        axes[1, i].imshow(mask_np[z], cmap='viridis')
        axes[1, i].set_title("Máscara real")
        axes[2, i].imshow(pred_np[z], cmap='viridis')
        axes[2, i].set_title("Predicción")
    for ax in axes.ravel():
      ax.axis('off')
    plt.suptitle(title_prefix)
    plt.tight_layout()
    plt.show()

def overlay_segmentation(img_np, mask_np, pred_np, z=None, alpha=0.4):
    """
    Superpone GT y predicción sobre la imagen en un corte Z.

    Parámetros
    ----------
    img_np : np.ndarray
        Imagen (D, H, W).
    mask_np : np.ndarray
        Máscara real (D, H, W).
    pred_np : np.ndarray
        Predicción (D, H, W).
    z : int or None, opcional
        Índice Z del corte; por defecto mitad del volumen.
    alpha : float, opcional
        Transparencia de la superposición. Por defecto 0.4.

    Efecto
    ------
    Dibuja dos paneles: GT sobre imagen y Predicción sobre imagen.
    """

    if z is None:
        z = img_np.shape[0] // 2
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_np[z], cmap='gray')
    ax[0].imshow(mask_np[z], cmap='jet', alpha=alpha)
    ax[0].set_title("Máscara real sobre imagen")

    ax[1].imshow(img_np[z], cmap='gray')
    ax[1].imshow(pred_np[z], cmap='jet', alpha=alpha)
    ax[1].set_title("Predicción sobre imagen")

    for a in ax: a.axis('off')
    plt.suptitle(f"Corte Z={z}")
    plt.tight_layout()
    plt.show()

def qualitative_eval(model, dataset, device, num_classes=3, idx=None):
    """
    Pipeline visual: predice una muestra y muestra cortes, comparación y overlays.

    Parámetros
    ----------
    model : nn.Module
        Modelo de segmentación 3D.
    dataset : torch.utils.data.Dataset
        Devuelve (img, mask) con formas compatibles.
    device : str or torch.device
        Dispositivo de inferencia.
    num_classes : int, opcional
        1 para binaria; >1 para multiclase. Por defecto 3.
    idx : int or None, opcional
        Índice de la muestra; aleatorio si None.

    Efecto
    ------
    - Ejecuta inferencia sin gradientes.
    - Llama a `show_random_slice`, `compare_slices` y `overlay_segmentation`.
    - Imprime el índice mostrado.
    """

    model.eval()
    if idx is None:
        idx = np.random.randint(0, len(dataset))
    img, mask = dataset[idx]
    img = img.unsqueeze(0).to(device)
    mask_np = mask.cpu().numpy()

    with torch.no_grad():
        logits = model(img)
        if num_classes == 1:
            pred = (torch.sigmoid(logits) > 0.5).float()
            pred_np = pred.squeeze().cpu().numpy()
        else:
            pred = logits.argmax(dim=1).squeeze()
            pred_np = pred.cpu().numpy()

    img_np = img.squeeze().cpu().numpy()

    print(f"Mostrando ejemplo #{idx}")
    show_random_slice(model, dataset, device, num_classes=num_classes, idx=idx)
    compare_slices(mask_np, pred_np, img_np, title_prefix=f"Volumen {idx}")
    overlay_segmentation(img_np, mask_np, pred_np)