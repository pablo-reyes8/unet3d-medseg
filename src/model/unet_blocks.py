import torch.nn as nn
import torch.nn.functional as F


############## U-NET3D UTILS #################

class ConvRelu3d(nn.Module):
    """
    Bloque convolucional 3D con normalización opcional y activación ReLU.

    Combina una capa Conv3d con BatchNorm3d o InstanceNorm3d (opcional),
    seguida de una función de activación ReLU o LeakyReLU.

    Parámetros
    ----------
    in_c : int
        Número de canales de entrada.
    out_c : int
        Número de canales de salida.
    k : int, opcional
        Tamaño del kernel de convolución. Por defecto 3.
    s : int, opcional
        Stride de la convolución. Por defecto 1.
    p : int, opcional
        Padding aplicado. Por defecto 1 (modo “same-like”).
    bias : bool, opcional
        Si se usa sesgo (bias) en la convolución. Por defecto True.
    norm : {'bn', 'in', None}, opcional
        Tipo de normalización a aplicar:
        - 'bn': Batch Normalization.
        - 'in': Instance Normalization.
        - None: sin normalización.
        Por defecto 'in'.
    act : {'relu', 'leaky_relu'}, opcional
        Tipo de activación no lineal. Por defecto 'relu'.

    Métodos
    -------
    forward(x):
        Aplica la secuencia Conv3D → (Norm) → (Activación) sobre el tensor `x`.

    Notas
    -----
    - Los pesos de la convolución se inicializan con Kaiming Normal según ReLU.
    - Diseñado como bloque base para arquitecturas 3D tipo U-Net o V-Net.
    """

    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=True, norm='in', act='relu'):
        super().__init__()
        layers = [nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)]
        if norm == 'bn':
            layers.append(nn.BatchNorm3d(out_c))
        elif norm == 'in':
            layers.append(nn.InstanceNorm3d(out_c, affine=True))
        if act == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif act == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.01, inplace=True))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class MaxPool3d(nn.Module):
    """
    Capa de MaxPooling tridimensional para volúmenes (D, H, W).

    Aplica una operación de reducción de resolución sobre cada canal
    del volumen, manteniendo el número de canales.

    Parámetros
    ----------
    k : int, opcional
        Tamaño del kernel de pooling. Por defecto 2.
    s : int, opcional
        Stride del pooling. Por defecto 2.
    p : int, opcional
        Padding aplicado antes del pooling. Por defecto 0.

    Métodos
    -------
    forward(x):
        Aplica la operación de MaxPooling 3D sobre el tensor `x`.

    Retorna
    -------
    torch.Tensor
        Tensor con la misma cantidad de canales, pero con dimensiones espaciales reducidas.
    """
    def __init__(self, k=2, s=2, p=0):
        super().__init__()
        self.net = nn.MaxPool3d(kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        return self.net(x)


class UpConv3d(nn.Module):
    """
    Capa de upsampling tridimensional mediante convolución transpuesta.

    Aumenta la resolución espacial y de profundidad de un volumen 3D,
    comúnmente usada en las etapas de decodificación de redes tipo U-Net.

    Parámetros
    ----------
    in_c : int
        Número de canales de entrada (nivel más profundo).
    out_c : int
        Número de canales de salida tras la convolución transpuesta.
    k : int, opcional
        Tamaño del kernel. Por defecto 2.
    s : int, opcional
        Stride de la convolución. Por defecto 2.
    p : int, opcional
        Padding aplicado. Por defecto 0.

    Métodos
    -------
    forward(x):
        Aplica la convolución transpuesta 3D para realizar el upsampling del tensor `x`.

    Retorna
    -------
    torch.Tensor
        Tensor con resolución aumentada y `out_c` canales.
    """
    def __init__(self, in_c, out_c, k=2, s=2, p=0):
        super().__init__()
        self.net = nn.ConvTranspose3d(in_c, out_c, kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        return self.net(x)