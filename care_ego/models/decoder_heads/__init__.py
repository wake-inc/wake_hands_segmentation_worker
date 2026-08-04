"""Decoder heads."""

from .add_Unet_decoder_with_seperate_heads_obj import CaregoDecoder3
from .add_Unet_deocder_output import CaregoDecoder
from .add_unet_deocder_input import CaregoDecoder2

__all__ = ["CaregoDecoder", "CaregoDecoder2", "CaregoDecoder3"]
