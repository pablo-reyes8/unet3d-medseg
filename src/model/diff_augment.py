import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

def _affine_grid_3d(theta: torch.Tensor, size):
    """
    Construye una grilla de muestreo 3D para una transformación afín.

    Parámetros
    ----------
    theta : torch.Tensor
        Parámetros afines por batch con forma (B, 3, 4).
    size : tuple
        Tamaño del tensor objetivo (B, C, D, H, W).

    Retorna
    -------
    torch.Tensor
        Grilla de coordenadas normalizadas con forma (B, D, H, W, 3), lista para grid_sample.
    """
    B, C, D, H, W = size
    return F.affine_grid(theta, size=(B, C, D, H, W), align_corners=False)

def _apply_grid_sample_3d(x: torch.Tensor, grid: torch.Tensor, mode='bilinear'):
    """
    Aplica muestreo en grilla 3D (warp) a un volumen.

    Parámetros
    ----------
    x : torch.Tensor
        Volumen de entrada (B, C, D, H, W).
    grid : torch.Tensor
        Grilla de coordenadas (B, D, H, W, 3) en [-1, 1].
    mode : {'bilinear', 'nearest'}, opcional
        Interpolación utilizada. Por defecto 'bilinear'.

    Retorna
    -------
    torch.Tensor
        Volumen deformado con el mismo tamaño que `x`.
    """
    return F.grid_sample(x, grid, mode=mode, padding_mode='border', align_corners=False)

def rand_brightness_contrast_3d(x, p=0.5, b_range=(-0.1, 0.1), c_range=(0.9, 1.1)):
    """
    Aumento aleatorio de brillo y contraste sobre volúmenes 3D en [0, 1].

    Parámetros
    ----------
    x : torch.Tensor
        Tensor (B, C, D, H, W) con intensidades en [0, 1].
    p : float, opcional
        Probabilidad de aplicar la transformación. Por defecto 0.5.
    b_range : tuple(float, float), opcional
        Rango uniforme para brillo (offset). Por defecto (-0.1, 0.1).
    c_range : tuple(float, float), opcional
        Rango uniforme para contraste (escala). Por defecto (0.9, 1.1).

    Retorna
    -------
    torch.Tensor
        Tensor aumentado, recortado a [0, 1].
    """
    if torch.rand(1).item() < p:
        b = torch.empty(x.size(0), 1, 1, 1, 1, device=x.device).uniform_(*b_range)
        c = torch.empty(x.size(0), 1, 1, 1, 1, device=x.device).uniform_(*c_range)
        x = x * c + b
    return x.clamp_(0, 1)

def rand_gamma_3d(x, p=0.5, gamma_range=(0.8, 1.25), eps=1e-6):
    """
    Corrección gamma aleatoria sobre volúmenes 3D en [0, 1].

    Parámetros
    ----------
    x : torch.Tensor
        Tensor (B, C, D, H, W) con intensidades en [0, 1].
    p : float, opcional
        Probabilidad de aplicar la transformación. Por defecto 0.5.
    gamma_range : tuple(float, float), opcional
        Rango uniforme para el exponente gamma. Por defecto (0.8, 1.25).
    eps : float, opcional
        Estabilidad numérica antes de elevar a potencia. Por defecto 1e-6.

    Retorna
    -------
    torch.Tensor
        Tensor aumentado, recortado a [0, 1].
    """
    if torch.rand(1).item() < p:
        g = torch.empty(x.size(0), 1, 1, 1, 1, device=x.device).uniform_(*gamma_range)
        x = torch.clamp(x, 0, 1)
        x = torch.pow(x + eps, g)
    return x.clamp_(0, 1)

def rand_gauss_noise_3d(x, p=0.3, sigma=(0.0, 0.02)):
    """
    Añade ruido gaussiano aleatorio a volúmenes 3D.

    Parámetros
    ----------
    x : torch.Tensor
        Tensor (B, C, D, H, W) con intensidades (idealmente en [0, 1]).
    p : float, opcional
        Probabilidad de aplicar ruido. Por defecto 0.3.
    sigma : tuple(float, float), opcional
        Rango uniforme para la desviación estándar del ruido. Por defecto (0.0, 0.02).

    Retorna
    -------
    torch.Tensor
        Tensor con ruido añadido, recortado a [0, 1].
    """
    if torch.rand(1).item() < p:
        std = torch.empty(x.size(0), 1, 1, 1, 1, device=x.device).uniform_(*sigma)
        noise = torch.randn_like(x) * std
        x = x + noise
    return x.clamp_(0, 1)

def rand_flip_3d(x, y, p=0.5):
    """
    Invierte aleatoriamente los volúmenes a lo largo de ejes D/H/W.

    Parámetros
    ----------
    x : torch.Tensor
        Imagen (B, C, D, H, W).
    y : torch.Tensor
        Máscara/etiquetas (B, D, H, W) alineadas con `x`.
    p : float, opcional
        Probabilidad global de aplicar flips (cada eje se evalúa aparte). Por defecto 0.5.

    Retorna
    -------
    tuple(torch.Tensor, torch.Tensor)
        (x_flip, y_flip) tras posibles volteos a lo largo de cada eje.
    """
    if torch.rand(1).item() < p:
        if torch.rand(1).item() < 0.33:
            x = torch.flip(x, dims=[2]); y = torch.flip(y, dims=[1])
        if torch.rand(1).item() < 0.33:
            x = torch.flip(x, dims=[3]); y = torch.flip(y, dims=[2])
        if torch.rand(1).item() < 0.33:
            x = torch.flip(x, dims=[4]); y = torch.flip(y, dims=[3])
    return x, y

def rand_affine_small_3d(x, y, p=0.35, max_rot=10, max_trans=0.05, max_scale=0.10):
    """
    Deformación afín pequeña y coherente (imagen+máscara) en 3D.

    Aplica rotaciones (±max_rot grados), traslaciones (±max_trans del tamaño)
    y escalado (±max_scale). Interpola imagen en bilinear y máscara en nearest.

    Parámetros
    ----------
    x : torch.Tensor
        Imagen (B, C, D, H, W) con intensidades en [0, 1].
    y : torch.Tensor
        Máscara/etiquetas (B, D, H, W).
    p : float, opcional
        Probabilidad de aplicar la transformación. Por defecto 0.35.
    max_rot : float, opcional
        Rotación máxima en grados por eje. Por defecto 10.
    max_trans : float, opcional
        Traslación máxima relativa por eje (en [-1, 1] coords). Por defecto 0.05.
    max_scale : float, opcional
        Factor máximo de escala relativo (±). Por defecto 0.10.

    Retorna
    -------
    tuple(torch.Tensor, torch.Tensor)
        (x_aug, y_aug): imagen y máscara deformadas afínmente.
    """
    if torch.rand(1).item() >= p:
        return x, y

    B, C, D, H, W = x.shape
    device = x.device

    rot_x = (torch.rand(B, device=device) * 2 - 1) * math.radians(max_rot)
    rot_y = (torch.rand(B, device=device) * 2 - 1) * math.radians(max_rot)
    rot_z = (torch.rand(B, device=device) * 2 - 1) * math.radians(max_rot)
    scale = 1.0 + (torch.rand(B, device=device) * 2 - 1) * max_scale
    tx = (torch.rand(B, device=device) * 2 - 1) * max_trans
    ty = (torch.rand(B, device=device) * 2 - 1) * max_trans
    tz = (torch.rand(B, device=device) * 2 - 1) * max_trans

    thetas = []
    for b in range(B):
        sx = sy = sz = scale[b]
        cx, cy, cz = torch.cos(rot_x[b]), torch.cos(rot_y[b]), torch.cos(rot_z[b])
        sxn, syn, szn = torch.sin(rot_x[b]), torch.sin(rot_y[b]), torch.sin(rot_z[b])

        Rz = torch.tensor([[cz, -szn, 0.0],[szn,  cz, 0.0],[0.0, 0.0, 1.0]], device=device)
        Ry = torch.tensor([[cy, 0.0,  syn],[0.0, 1.0, 0.0],[-syn, 0.0, cy]], device=device)
        Rx = torch.tensor([[1.0, 0.0, 0.0],[0.0, cx, -sxn],[0.0, sxn,  cx]], device=device)

        R = Rz @ Ry @ Rx
        S = torch.diag(torch.tensor([sx, sy, sz], device=device))
        A = (R @ S)

        theta = torch.zeros(3, 4, device=device)
        theta[:, :3] = A
        theta[:, 3]  = torch.tensor([tx[b], ty[b], tz[b]], device=device)
        thetas.append(theta)

    theta = torch.stack(thetas, dim=0)
    grid  = _affine_grid_3d(theta, size=(B, C, D, H, W))

    x_aug = _apply_grid_sample_3d(x, grid, mode='bilinear')
    y_in  = y.unsqueeze(1).float()
    y_aug = _apply_grid_sample_3d(y_in, grid, mode='nearest').squeeze(1).long()
    return x_aug, y_aug


class DiffAugment3D:
    """
    Aumentos de datos 3D ligeros para escenarios con pocos ejemplos.

    Incluye:
    - Intensidad: brillo/contraste, gamma y ruido gaussiano.
    - Espaciales: flips 3D y transformación afín pequeña (rot/trasl/escala).

    Parámetros
    ----------
    p_flip : float, opcional
        Probabilidad de aplicar flips 3D. Por defecto 0.5.
    p_affine : float, opcional
        Probabilidad de aplicar una afín pequeña. Por defecto 0.35.
    use_brightness_contrast : bool, opcional
        Activar ajuste aleatorio de brillo/contraste. Por defecto True.
    use_gamma : bool, opcional
        Activar corrección gamma aleatoria. Por defecto True.
    use_noise : bool, opcional
        Activar ruido gaussiano aleatorio. Por defecto True.

    Llamada
    -------
    __call__(xb, yb)
        Aplica los aumentos de forma coherente (imagen+máscara).

    Retorna
    -------
    tuple(torch.Tensor, torch.Tensor)
        (xb_aug, yb_aug) con mismas formas que la entrada:
        - xb: (B, 1, D, H, W) en [0, 1]
        - yb: (B, D, H, W) en {0..C-1}

    Notas
    -----
    - No calcula gradientes (decorador @torch.no_grad()).
    - Usa:
        rand_flip_3d → rand_affine_small_3d → (opcionales) brightness/contrast, gamma, ruido.
    """
    def __init__(self,p_flip=0.5, p_affine=0.35,
                 use_brightness_contrast=True,
                 use_gamma=True,
                 use_noise=True):

        self.p_flip = p_flip
        self.p_affine = p_affine
        self.use_bc = use_brightness_contrast
        self.use_gamma = use_gamma
        self.use_noise = use_noise

    @torch.no_grad()
    def __call__(self, xb: torch.Tensor, yb: torch.Tensor):
        """
        xb : torch.Tensor
            Imagen (B, 1, D, H, W) en [0, 1].
        yb : torch.Tensor
            Máscara (B, D, H, W) con etiquetas enteras.
        """
        xb, yb = rand_flip_3d(xb, yb, p=self.p_flip)
        xb, yb = rand_affine_small_3d(xb, yb, p=self.p_affine)

        if self.use_bc:   xb = rand_brightness_contrast_3d(xb, p=0.7)
        if self.use_gamma: xb = rand_gamma_3d(xb, p=0.6)
        if self.use_noise: xb = rand_gauss_noise_3d(xb, p=0.3)

        return xb, yb