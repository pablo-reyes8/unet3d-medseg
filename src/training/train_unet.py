import torch 
from src.training.train_one_epoch import *

def train_uneted(
    model, optimizer, device, criterion, num_classes, epocs, train_loader ,val_loader , 
    patience=10, min_delta=0.0, augmnet=None,
    target_metric=None,
    save_best_path=None):

    """
    Entrena U-Net 3D con early stopping sobre la métrica de validación (Dice o mIoU).

    Parámetros
    ----------
    model : nn.Module
        Modelo de segmentación 3D (produce logits B×C×D×H×W).
    optimizer : torch.optim.Optimizer
        Optimizador para actualizar los parámetros del modelo.
    device : str or torch.device
        Dispositivo donde correr (e.g., "cuda" o "cpu").
    criterion : callable
        Función de pérdida (binaria o multiclase).
    num_classes : int
        1 → binaria (métrica: Dice); >1 → multiclase (métrica: mIoU).
    epocs : int
        Número máximo de épocas de entrenamiento.
    patience : int, opcional
        Épocas sin mejora antes de detener (early stopping). Por defecto 10.
    min_delta : float, opcional
        Mejora mínima requerida para resetear la paciencia. Por defecto 0.0.
    augmnet : callable or None, opcional
        Función de aumentación coherente (xb, yb) → (xb, yb). Por defecto None.
    target_metric : float or None, opcional
        Umbral de métrica para detener de forma anticipada si se alcanza. Por defecto None.
    save_best_path : str or None, opcional
        Ruta para guardar el mejor checkpoint (estado del modelo+optimizador). Por defecto None.

    Retorna
    -------
    tuple(dict, dict)
        history_train, history_val — diccionarios con métricas por época.

    Comportamiento
    --------------
    - Selecciona métrica clave: "Dice" (binaria) o "mIoU" (multiclase).
    - Guarda el mejor estado cuando hay mejora > min_delta.
    - Detiene si:
      * No hay mejora por `patience` épocas, o
      * Se alcanza `target_metric`.
    - Al final, restaura el mejor checkpoint si existe.
    """

    history_train, history_val = {}, {}
    metric_key = "mIoU" if num_classes > 1 else "Dice"

    best_metric = float("-inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epocs + 1):
        tr = train_epoch_seg_3d(
            train_loader, model, optimizer, criterion,
            augment_fn=augmnet, num_classes=num_classes,
            device=device, amp=False, desc=f"Train {epoch}")

        va = eval_epoch_seg_3d(
            val_loader, model, criterion,
            num_classes=num_classes, device=device, desc=f"Val  {epoch}")

        history_train[f"Epoch {epoch}"] = tr
        history_val[f"Epoch {epoch}"] = va

        val_metric = va.get(metric_key, None)
        if val_metric is None or (isinstance(val_metric, float) and val_metric != val_metric):
            current_metric = float("-inf")
        else:
            current_metric = float(val_metric)

        improved = (current_metric - best_metric) > float(min_delta)

        if improved:
            best_metric = current_metric
            best_state = {
                "epoch": epoch,
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "best_metric": best_metric,
                "metric_key": metric_key}
            epochs_no_improve = 0

            if save_best_path is not None:
                torch.save(best_state, save_best_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"[EarlyStop] Sin mejora en {metric_key} por {patience} épocas. Stop en epoch {epoch}.")
            break

        if target_metric is not None and current_metric >= float(target_metric):
            print(f"[TargetReached] {metric_key}={current_metric:.4f} ≥ {float(target_metric):.4f}. Stop en epoch {epoch}.")
            # Asegura guardar si este epoch es el mejor hasta ahora
            if best_state is None or current_metric > best_state.get("best_metric", float("-inf")):
                best_state = {"epoch": epoch,
                    "model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "best_metric": current_metric,
                    "metric_key": metric_key}
                
                if save_best_path is not None:
                    torch.save(best_state, save_best_path)
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        if save_best_path is not None:
            print(f"[Info] Mejor checkpoint restaurado del epoch {best_state['epoch']} "
                  f"({metric_key}={best_state['best_metric']:.4f}).")
    else:
        print("[Warn] No se registró una métrica válida; el modelo queda con el último estado entrenado.")

    return history_train, history_val