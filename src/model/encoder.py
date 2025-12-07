from src.model.unet_blocks import *

class UnetEncoderLayer3d(nn.Module):
    """
    Bloque codificador (encoder) de una U-Net 3D.

    Aplica dos convoluciones 3D con normalización y activación ReLU,
    seguidas opcionalmente de dropout y max pooling.
    Devuelve tanto la salida reducida (`x_down`) como la característica intermedia (`skip`)
    para la conexión con el decodificador.

    Parámetros
    ----------
    in_c : int
        Número de canales de entrada.
    out_c : int
        Número de canales de salida del bloque.
    norm : {'in', 'bn', None}, opcional
        Tipo de normalización a usar en las convoluciones. Por defecto 'in'.
    dropout : float, opcional
        Probabilidad de dropout tras las convoluciones (0 = desactivado). Por defecto 0.0.
    use_pool : bool, opcional
        Si True, aplica MaxPool3d al final del bloque. Por defecto True.

    Métodos
    -------
    forward(x):
        Ejecuta el bloque y retorna:
        - x_down : torch.Tensor — salida reducida tras pooling/dropout.
        - skip   : torch.Tensor — salida previa al pooling, usada como conexión lateral.

    Retorna
    -------
    tuple(torch.Tensor, torch.Tensor)
        (x_down, skip), donde `x_down` se usa como entrada del siguiente nivel del encoder
        y `skip` se concatena con el correspondiente bloque del decoder.
    """
    def __init__(self, in_c, out_c, norm='in', dropout=0.0, use_pool=True):
        super().__init__()
        self.conv1 = ConvRelu3d(in_c,  out_c, norm=norm)
        self.conv2 = ConvRelu3d(out_c, out_c, norm=norm)
        self.dropout = nn.Dropout3d(dropout) if dropout and dropout > 0 else nn.Identity()
        self.pool = MaxPool3d(k=2, s=2) if use_pool else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x          # guardar para skip-connection
        x = self.dropout(x)
        x = self.pool(x)
        return x, skip