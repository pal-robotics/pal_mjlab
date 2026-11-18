import torch
import torch.nn as nn

class BaseModel(nn.Module) :
    def __init__(self):
        pass

class EncoderModel (BaseModel):
    def __init__(self):
        super().__init__()

class DecoderModel (BaseModel):
    def __init__(self):
        super().__init__()

class CriticModel (BaseModel):
    def __init__(self):
        super().__init__()