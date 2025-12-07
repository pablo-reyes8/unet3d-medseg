from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from contextlib import nullcontext

def get_autocast_ctx(enabled: bool):
    """
    Obtiene un context manager de autocast para CUDA si está habilitado.

    Parámetros
    ----------
    enabled : bool
        Si False, retorna un no-op (nullcontext). Si True, intenta usar torch.amp
        y, en caso de fallo, torch.cuda.amp.

    Retorna
    -------
    contextmanager
        Contexto de autocast (o nullcontext si disabled).
    """
    if not enabled:
        return nullcontext()
    try:
        from torch.amp import autocast as _autocast_new
        return _autocast_new(device_type="cuda", enabled=True)

    except Exception:
        from torch.cuda.amp import autocast as _autocast_old
        return _autocast_old(enabled=True)

def make_scaler(enabled: bool):
    """
    Crea un GradScaler para AMP si está disponible.

    Parámetros
    ----------
    enabled : bool
        Si True, intenta instanciar GradScaler (torch.amp o torch.cuda.amp).

    Retorna
    -------
    tuple
        (scaler, amp_ok):
        - scaler : GradScaler o None.
        - amp_ok : bool indicando si AMP quedó habilitado.
    """
    if not enabled:
        return None, False

    try:
        from torch.amp import GradScaler as _GradScalerNew
        try:
            scaler = _GradScalerNew("cuda", enabled=True)
        except TypeError:
            scaler = _GradScalerNew(enabled=True)
        return scaler, True
    except Exception:
        try:
            from torch.cuda.amp import GradScaler as _GradScalerOld
            scaler = _GradScalerOld(enabled=True)
            return scaler, True
        except Exception:
            print("[AMP] GradScaler no disponible; continúo en FP32.")
            return None, False