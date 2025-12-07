from src.model.unet_blocks import *
import torch

def _match_spatial_3d(x, ref):
    """
    Ajusta el tamaño espacial de un tensor 3D para que coincida con otro tensor de referencia.

    Si las dimensiones (D, H, W) de `x` difieren de las de `ref`, se aplica:
    - Recorte centrado si `x` es más grande.
    - Padding centrado (reflect o constante) si `x` es más pequeño.

    Parámetros
    ----------
    x : torch.Tensor
        Tensor de entrada con forma (B, C, D, H, W).
    ref : torch.Tensor
        Tensor de referencia con la forma deseada (B, C, D_ref, H_ref, W_ref).

    Retorna
    -------
    torch.Tensor
        Tensor ajustado para tener las mismas dimensiones espaciales que `ref`.

    Notas
    -----
    - El recorte y el padding son simétricos (centrados).
    - Se usa `torch.nn.functional.pad` para expandir el tensor si es necesario.
    - Los canales (C) y el batch (B) no se alteran.
    """
    _, _, D, H, W = x.shape
    _, _, Dr, Hr, Wr = ref.shape

    dD, dH, dW = D - Dr, H - Hr, W - Wr

    if dD > 0:
        s = dD // 2
        x = x[:, :, s:D - (dD - s), :, :]
        D = x.shape[2]
    if dH > 0:
        s = dH // 2
        x = x[:, :, :, s:H - (dH - s), :]
        H = x.shape[3]
    if dW > 0:
        s = dW // 2
        x = x[:, :, :, :, s:W - (dW - s)]

    pads = []
    dW = Wr - x.shape[4]
    dH = Hr - x.shape[3]
    dD = Dr - x.shape[2]

    for d in (dW, dH, dD):
        if d > 0:
            left = d // 2
            right = d - left
            pads.extend([left, right])
        else:
            pads.extend([0, 0])

    if any(p > 0 for p in pads):
        x = F.pad(x, (pads[0], pads[1], pads[2], pads[3], pads[4], pads[5]))
    return x


class UnetDecoderLayer3d(nn.Module):
    """
    Bloque decodificador (decoder) de una U-Net 3D.

    Realiza upsampling mediante convolución transpuesta, concatena con la salida
    del encoder correspondiente (skip-connection) y aplica dos convoluciones 3D con
    normalización y activación ReLU.

    Parámetros
    ----------
    in_c : int
        Número de canales de entrada desde el nivel más profundo del decoder.
    skip_c : int
        Número de canales provenientes del skip-connection del encoder.
    out_c : int
        Número de canales de salida del bloque.
    norm : {'in', 'bn', None}, opcional
        Tipo de normalización a aplicar en las convoluciones. Por defecto 'in'.
    dropout : float, opcional
        Probabilidad de dropout tras la primera convolución. Por defecto 0.0.

    Métodos
    -------
    forward(x, skip):
        - Aplica la convolución transpuesta para duplicar la resolución (D, H, W).
        - Ajusta el tamaño de `x` para coincidir con `skip` mediante `_match_spatial_3d`.
        - Concatena ambos tensores en la dimensión de canales.
        - Aplica dos convoluciones 3D (Conv3d + ReLU), con dropout opcional.

    Retorna
    -------
    torch.Tensor
        Tensor decodificado con resolución aumentada y `out_c` canales.

    Notas
    -----
    Este bloque corresponde a una etapa de “upsampling” en la U-Net 3D
    y permite la fusión entre la información de alto nivel (profunda)
    y los detalles espaciales preservados por el encoder.
    """
    def __init__(self, in_c, skip_c, out_c, norm='in', dropout=0.0):
        super().__init__()
        self.up    = UpConv3d(in_c, out_c, k=2, s=2, p=0)
        self.conv1 = ConvRelu3d(out_c + skip_c, out_c, norm=norm)
        self.conv2 = ConvRelu3d(out_c, out_c, norm=norm)
        self.dropout = nn.Dropout3d(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)

        # Alinear espacialmente con el skip
        x = _match_spatial_3d(x, skip)

        # Concat en canales y doble conv
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x