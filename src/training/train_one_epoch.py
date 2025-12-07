import torch 
from src.training.metrics import *
from src.training.autocast import *

def train_epoch_seg_3d(
    dataloader, model, optimizer, criterion,
    num_classes=1, device=None, amp=False, desc="Train" , augment_fn=None):
    """
    Entrena un epoch de segmentación 3D (binaria o multiclase) con métricas básicas.

    Parámetros
    ----------
    dataloader : torch.utils.data.DataLoader
        Lotes (xb, yb) donde xb∈[0,1] con forma (B, 1, D, H, W) y yb con forma (B, D, H, W).
    model : nn.Module
        Red de segmentación 3D que produce logits (B, C, D, H, W).
    optimizer : torch.optim.Optimizer
        Optimizador para actualizar parámetros del modelo.
    criterion : callable
        Función de pérdida. Para binaria: recibe (logits, target_bin canalizado).
        Para multiclase: recibe (logits, target_long).
    num_classes : int, opcional
        1 para segmentación binaria (usa sigmoid + Dice); >1 para multiclase (usa argmax + mIoU).
    device : str or torch.device, opcional
        Dispositivo destino; por defecto usa el del modelo.
    amp : bool, opcional
        Activa Automatic Mixed Precision (CUDA) si disponible. Por defecto False.
    desc : str, opcional
        Texto para la barra de progreso. Por defecto "Train".
    augment_fn : callable or None, opcional
        Función de aumentación coherente (xb, yb)→(xb, yb). Por defecto None.

    Retorna
    -------
    dict
        {'loss': float, 'vox_acc': float, 'Dice'| 'mIoU': float}
        - 'vox_acc' es exactitud voxel a voxel (%).
        - Usa 'Dice' si num_classes==1; 'mIoU' si num_classes>1.

    Notas
    -----
    - En binaria, el target se convierte a [B,1,D,H,W] y se umbraliza predicción en 0.5.
    - AMP se gestiona con GradScaler cuando CUDA está disponible.
    """

    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    want_amp = bool(amp and torch.cuda.is_available() and device.type == "cuda")
    scaler, use_amp = make_scaler(want_amp)
    autocast_ctx = get_autocast_ctx(use_amp)

    model.train()
    running_loss = 0.0
    running_metric = 0.0
    correct_vox = 0
    n_vox = 0
    n_samples = 0

    pbar = tqdm(dataloader, total=len(dataloader), leave=False, desc=desc)
    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)


        # Diff Augment
        if augment_fn is not None:
            xb, yb = augment_fn(xb, yb)

        optimizer.zero_grad(set_to_none=True)

        # Forward Pass
        with autocast_ctx:
            logits = model(xb)
            if num_classes == 1:
                yb_bin = ensure_binary_target_3d(yb)
                loss = criterion(logits, yb_bin)
            else:
                loss = criterion(logits, yb.long())


        # Back propagation and optimizer step
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = xb.size(0)
        running_loss += float(loss.detach().cpu()) * bs
        n_samples += bs

        # Segmentation Metrics
        with torch.no_grad():
            if num_classes == 1:
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()
                yb_bin = ensure_binary_target_3d(yb)

                correct_vox += (pred == yb_bin).sum().item()
                n_vox       += yb_bin.numel()

                dice = dice_coeff_3d(pred, yb_bin)
                running_metric += float(dice.cpu()) * bs
                metric_name = "Dice"
            else:
                pred = logits.argmax(dim=1)
                correct_vox += (pred == yb).sum().item()
                n_vox       += yb.numel()

                miou = mean_iou_mc_3d(pred, yb, num_classes)
                running_metric += float(miou) * bs
                metric_name = "mIoU"

        vox_acc = 100.0 * correct_vox / max(1, n_vox)
        pbar.set_postfix(
            loss=f"{running_loss/max(1,n_samples):.4f}",
            vox_acc=f"{vox_acc:.2f}%",
            **{metric_name: f"{running_metric/max(1,n_samples):.3f}"})


    epoch_loss = running_loss / max(1, n_samples)
    vox_acc    = 100.0 * correct_vox / max(1, n_vox)
    final_metric = running_metric / max(1, n_samples)
    print(f"{desc} - loss: {epoch_loss:.4f} | vox_acc: {vox_acc:.2f}% | {metric_name}: {final_metric:.3f}")
    return {'loss': epoch_loss, 'vox_acc': vox_acc, metric_name: final_metric}



@torch.no_grad()
def eval_epoch_seg_3d(dataloader, model, criterion, num_classes=1, device=None, desc="Val"):
    """
    Evalúa un epoch de segmentación 3D (binaria o multiclase) sin gradientes.

    Parámetros
    ----------
    dataloader : torch.utils.data.DataLoader
        Lotes (xb, yb) donde xb∈[0,1] con forma (B, 1, D, H, W) y yb con forma (B, D, H, W).
    model : nn.Module
        Red de segmentación 3D que produce logits (B, C, D, H, W).
    criterion : callable
        Función de pérdida. Binaria: (logits, target_bin canalizado). Multiclase: (logits, target_long).
    num_classes : int, opcional
        1 para binaria (usa sigmoid, Dice); >1 para multiclase (argmax, mIoU). Por defecto 1.
    device : str or torch.device, opcional
        Dispositivo a usar; por defecto el del modelo.
    desc : str, opcional
        Etiqueta para la barra de progreso. Por defecto "Val".

    Métricas
    --------
    - loss promedio (criterion)
    - vox_acc : exactitud voxel a voxel (%)
    - Dice (si num_classes == 1) o mIoU (si num_classes > 1)

    Retorna
    -------
    dict
        {'loss': float, 'vox_acc': float, 'Dice' | 'mIoU': float}

    Notas
    -----
    - No actualiza pesos (decorador @torch.no_grad).
    - En binaria, el target se adapta a [B,1,D,H,W] y el umbral es 0.5 sobre sigmoid(logits).
    """
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    model.eval()
    running_loss = 0.0
    running_metric = 0.0
    correct_vox = 0
    n_vox = 0
    n_samples = 0

    pbar = tqdm(dataloader, total=len(dataloader), leave=False, desc=desc)
    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        bs = xb.size(0)
        n_samples += bs

        if num_classes == 1:
            yb_bin = ensure_binary_target_3d(yb)
            loss = criterion(logits, yb_bin)
        else:
            loss = criterion(logits, yb.long())
        running_loss += float(loss.detach().cpu()) * bs

        if num_classes == 1:
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
            yb_bin = ensure_binary_target_3d(yb)

            correct_vox += (pred == yb_bin).sum().item()
            n_vox       += yb_bin.numel()

            dice = dice_coeff_3d(pred, yb_bin)
            running_metric += float(dice.cpu()) * bs
            metric_name = "Dice"
        else:
            pred = logits.argmax(dim=1)
            correct_vox += (pred == yb).sum().item()
            n_vox       += yb.numel()

            miou = mean_iou_mc_3d(pred, yb, num_classes)
            running_metric += float(miou) * bs
            metric_name = "mIoU"

        vox_acc = 100.0 * correct_vox / max(1, n_vox)
        pbar.set_postfix(loss=f"{running_loss/max(1,n_samples):.4f}",
            vox_acc=f"{vox_acc:.2f}%",
            **{metric_name: f"{running_metric/max(1,n_samples):.3f}"})

    epoch_loss = running_loss / max(1, n_samples)
    vox_acc    = 100.0 * correct_vox / max(1, n_vox)
    final_metric = running_metric / max(1, n_samples)
    print(f"{desc} - loss: {epoch_loss:.4f} | vox_acc: {vox_acc:.2f}% | {metric_name}: {final_metric:.3f}")

    return {'loss': epoch_loss, 'vox_acc': vox_acc, metric_name: final_metric}