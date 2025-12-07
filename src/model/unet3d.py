import torch 
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoder import * 
from src.model.decoder import *

class UNet3D(nn.Module):
    """
    Implementación de una arquitectura U-Net 3D para segmentación volumétrica.

    La red sigue la estructura clásica encoder–decoder con conexiones tipo skip entre
    niveles simétricos. Está diseñada para tareas de segmentación médica en 3D
    (por ejemplo, MRI o CT), donde la entrada y la salida son volúmenes.

    Parámetros
    ----------
    in_channels : int, opcional
        Número de canales de entrada (por defecto 1, para imágenes en escala de grises).
    num_classes : int, opcional
        Número de clases de salida (canales del mapa de segmentación). Por defecto 3.
    base : int, opcional
        Número base de canales en el primer nivel del encoder. Por defecto 32.
        Los siguientes niveles escalan multiplicando por 2 en cada etapa.
    norm : {'in', 'bn', None}, opcional
        Tipo de normalización usada en las convoluciones. Por defecto 'in'.
    dropout : float, opcional
        Tasa de dropout 3D aplicada en encoder y decoder. Por defecto 0.0.

    Estructura
    ----------
    Encoder:
        4 bloques `UnetEncoderLayer3d`, duplicando canales en cada nivel.
    Bottleneck:
        Dos capas `ConvRelu3d` que procesan la representación más profunda.
    Decoder:
        4 bloques `UnetDecoderLayer3d`, que reducen canales y aumentan resolución.
    Output head:
        Convolución final 1×1×1 para mapear a `num_classes`.

    Métodos
    -------
    forward(x):
        Ejecuta el paso completo de inferencia:
        - Codifica la entrada y guarda los skips.
        - Pasa por el bottleneck.
        - Decodifica concatenando los skips correspondientes.
        - Genera los logits de salida (sin softmax).

    Retorna
    -------
    torch.Tensor
        Tensor de salida con forma (B, num_classes, D, H, W), que representa
        los logits por clase para cada voxel del volumen.

    Ejemplo
    --------
    >>> model = UNet3D(in_channels=1, num_classes=3)
    >>> x = torch.randn(1, 1, 64, 64, 64)
    >>> out = model(x)
    >>> out.shape
    torch.Size([1, 3, 64, 64, 64])
    """
    def __init__(self, in_channels=1, num_classes=3, base=32, norm='in', dropout=0.0):
        super().__init__()
        C = base

        #  Encoder
        self.enc = nn.ModuleList([
            UnetEncoderLayer3d(in_channels, C,   norm=norm, dropout=dropout),  # 1  ->  C  (skip1) D,H,W
            UnetEncoderLayer3d(C, 2*C,  norm=norm, dropout=dropout),  # C  ->  2C
            UnetEncoderLayer3d(2*C,4*C,  norm=norm, dropout=dropout),  # 2C ->  4C
            UnetEncoderLayer3d(4*C, 8*C,  norm=norm, dropout=dropout),  # 4C ->  8C
        ])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvRelu3d(8*C,16*C, norm=norm),   # 8C  -> 16C
            ConvRelu3d(16*C, 16*C, norm=norm),  # 16C -> 16C
        )

        #  Decoder
        self.dec = nn.ModuleList([
            UnetDecoderLayer3d(16*C,8*C, 8*C,  norm=norm, dropout=dropout),
            UnetDecoderLayer3d(8*C, 4*C, 4*C,  norm=norm, dropout=dropout),
            UnetDecoderLayer3d(4*C, 2*C, 2*C,  norm=norm, dropout=dropout),
            UnetDecoderLayer3d(2*C, C, C,    norm=norm, dropout=dropout), ])

        #  Cabeza de salida
        self.out_conv = nn.Conv3d(C, num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        # Encoder
        for layer in self.enc:
            x, skip = layer(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for layer, skip in zip(self.dec, reversed(skips)):
            x = layer(x, skip)

        logits = self.out_conv(x)
        return logits