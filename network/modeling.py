from .backbone import (
    AFASeg
)


def AFASeg_S(num_classes=19, output_stride=8):
    return AFASeg.model_type('AFASeg_S', num_classes)


def AFASeg_XS(num_classes=19, output_stride=8):
    return AFASeg.model_type('AFASeg_XS', num_classes)


def AFASeg_XXS(num_classes=19, output_stride=8):
    return AFASeg.model_type('AFASeg_XXS', num_classes)
