from abc import abstractmethod
from ..buffer import *

class TimestepBlock(nn.Module):


    @abstractmethod
    def forward(self, x, emb, z):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):

    def forward(self, x, emb, context=None, z=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, z)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x