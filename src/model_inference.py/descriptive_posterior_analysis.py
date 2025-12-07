import torch 
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def _collect_preds_targets(model, dataloader, device, num_classes=3):
    """
    Recolecta predicciones y etiquetas verdaderas sobre un conjunto de datos.

    Parámetros
    ----------
    model : nn.Module
        Modelo de segmentación 3D (produce logits B×C×D×H×W).
    dataloader : torch.utils.data.DataLoader
        Iterador sobre lotes (xb, yb).
    device : str or torch.device
        Dispositivo de inferencia.
    num_classes : int, opcional
        Número de clases de salida. Por defecto 3 (usa argmax sobre logits).

    Retorna
    -------
    tuple(np.ndarray, np.ndarray)
        - P : predicciones concatenadas (N, D, H, W)
        - Y : etiquetas verdaderas concatenadas (N, D, H, W)

    Notas
    -----
    - No calcula gradientes (`@torch.no_grad()`).
    - Asume que el modelo devuelve logits sin softmax.
    - Para tareas binarias, el `num_classes` no altera el comportamiento
      (solo aplica argmax sobre el canal 1).
    """

    model.eval()
    preds, targs = [], []
    for xb, yb in dataloader:
        xb = xb.to(device); yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(1)
        preds.append(pred.cpu().numpy())
        targs.append(yb.cpu().numpy())
    P = np.concatenate(preds,  axis=0)
    Y = np.concatenate(targs,  axis=0)
    return P, Y

def plot_iou_per_class(model, val_loader, device, num_classes=3, class_names=None):
    """
    Calcula y grafica el IoU por clase en el conjunto de validación.

    Parámetros
    ----------
    model : nn.Module
        Modelo de segmentación 3D.
    val_loader : torch.utils.data.DataLoader
        DataLoader del conjunto de validación.
    device : str or torch.device
        Dispositivo de inferencia.
    num_classes : int, opcional
        Número total de clases. Por defecto 3.
    class_names : list[str] or None, opcional
        Nombres de las clases para el eje X; si None, usa "Clase 0..C-1".

    Efecto
    ------
    - Muestra un gráfico de barras con el IoU de cada clase.
    - Si alguna clase no aparece (union=0), su IoU se marca como NaN.

    Notas
    -----
    - Usa `_collect_preds_targets` para generar matrices de predicción (P) y ground truth (Y).
    - IoU = intersección / unión, calculado de forma binaria por clase.
    """

    P, Y = _collect_preds_targets(model, val_loader, device, num_classes)
    ious = []
    for c in range(num_classes):
        inter = np.logical_and(P==c, Y==c).sum()
        union = np.logical_or (P==c, Y==c).sum()
        ious.append( (inter/union) if union>0 else np.nan )
    names = class_names or [f"Clase {c}" for c in range(num_classes)]
    plt.figure(figsize=(6,4))
    plt.bar(range(num_classes), ious)
    plt.xticks(range(num_classes), names, rotation=0)
    plt.ylim(0,1)
    plt.ylabel("IoU")
    plt.title("IoU por clase (validación)")
    plt.show()

def plot_iou_along_slices(model, val_loader, device, num_classes=3):
    """
    Calcula y grafica el IoU promedio por corte (eje Z) en los volúmenes del conjunto de validación.

    Parámetros
    ----------
    model : nn.Module
        Modelo de segmentación 3D.
    val_loader : torch.utils.data.DataLoader
        DataLoader con los volúmenes de validación.
    device : str or torch.device
        Dispositivo de inferencia.
    num_classes : int, opcional
        Número de clases de salida. Por defecto 3.

    Descripción
    -----------
    - Evalúa el modelo sobre `val_loader` y obtiene predicciones (P) y etiquetas (Y).
    - Para cada corte Z, calcula el IoU promedio entre clases presentes en ese plano.
    - Si un corte no contiene ninguna clase válida, se marca como NaN.

    Efecto
    ------
    Muestra un gráfico de línea con el IoU promedio por corte (Z) para visualizar
    cómo varía el desempeño del modelo a lo largo de la profundidad del volumen.

    Notas
    -----
    - Útil para analizar consistencia espacial del modelo en 3D.
    - El eje X representa los índices de corte Z.
    - Los valores NaN (sin clases presentes) se omiten del gráfico.
    """

    P, Y = _collect_preds_targets(model, val_loader, device, num_classes)
    Z = P.shape[1]
    iou_z = np.zeros(Z); counts = np.zeros(Z)
    for z in range(Z):
        inter = (P[:,z]==Y[:,z]).astype(np.int32)
        iou_slice = []
        for c in np.unique(Y[:,z]):
            if c < 0: continue
            inter_c = np.logical_and(P[:,z]==c, Y[:,z]==c).sum()
            union_c = np.logical_or (P[:,z]==c, Y[:,z]==c).sum()
            if union_c>0: iou_slice.append(inter_c/union_c)

        if iou_slice:
            iou_z[z] = np.mean(iou_slice); counts[z]=1

    iou_z[counts==0] = np.nan
    plt.figure(figsize=(7,4))
    plt.plot(np.arange(Z), iou_z, marker='o', linewidth=2)
    plt.xlabel("Corte Z")
    plt.ylabel("IoU promedio")
    plt.title("IoU a lo largo del volumen (validación)")
    plt.ylim(0,1)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_training_curves(history_train, history_val, metric_key="mIoU"):
    """
    Grafica las curvas de pérdida y de la métrica principal durante el entrenamiento.

    Parámetros
    ----------
    history_train : dict
        Historial del entrenamiento, con claves tipo 'Epoch 1', 'Epoch 2', ...
        y valores dict con métricas ('loss', 'mIoU' o 'Dice', etc.).
    history_val : dict
        Historial de validación con el mismo formato que `history_train`.
    metric_key : str, opcional
        Métrica a graficar en la segunda subfigura ('mIoU' o 'Dice'). Por defecto "mIoU".

    Descripción
    -----------
    - Extrae las series ordenadas por época desde los diccionarios de historial.
    - Dibuja dos subgráficos:
        (1) Curva de pérdida (train vs val)
        (2) Evolución de la métrica principal (train vs val)
    - Permite evaluar convergencia, sobreajuste y estabilidad del entrenamiento.

    Notas
    -----
    - Se espera que las claves sigan el formato "Epoch X" para su ordenamiento correcto.
    - Los valores de las métricas deben ser numéricos (float).
    """

    def _series(h, key):
        keys = sorted(h.keys(), key=lambda s: int(s.split()[-1]))
        return [h[k][key] for k in keys]

    tr_loss = _series(history_train, "loss")
    va_loss = _series(history_val,   "loss")
    tr_met  = _series(history_train, metric_key)
    va_met  = _series(history_val,   metric_key)

    epochs = np.arange(1, len(tr_loss)+1)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, tr_loss, marker='o', label="Train")
    plt.plot(epochs, va_loss, marker='o', label="Val")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Curva de pérdida")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, tr_met, marker='o', label=f"Train {metric_key}")
    plt.plot(epochs, va_met, marker='o', label=f"Val {metric_key}")
    plt.xlabel("Época")
    plt.ylabel(metric_key)
    plt.title(f"Evolución {metric_key}")
    plt.ylim(0,1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()