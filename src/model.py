import torch.nn as nn

class MenmaEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        pass

class MenmaDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __init_state__(self, enc_outputs):
        pass
    
    def forward(self, X, state):
        pass


class MenmaEncoderDecoder(nn.Module):

    encoder = MenmaEncoder()

    decoder = MenmaDecoder()

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, X):
        pass