from classy_vision.generic.util import register_transform
from classy_vision.dataset.transforms import ClassyTransform
from typing import Any, Dict, Union
from PIL import Image
import torch


@register_transform("CropSides")
class CropSides(ClassyTransform):
    """
    Crop different amounts from each side of the image (left, right, top, bottom).
    """

    def __init__(self, crop_left: int, crop_right: int, crop_top: int, crop_bottom: int):
        """
        Args:
            crop_left (int): Pixels to crop from the left.
            crop_right (int): Pixels to crop from the right.
            crop_top (int): Pixels to crop from the top.
            crop_bottom (int): Pixels to crop from the bottom.
        """
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom

    def __call__(self, image: Union[Image.Image, torch.Tensor]):
        """
        Crop the image on all four sides.

        Args:
            image (PIL.Image or torch.Tensor): The image to crop.

        Returns:
            Cropped image in the same format as input.
        """
        if isinstance(image, Image.Image):
            width, height = image.size
            left = self.crop_left
            upper = self.crop_top
            right = width - self.crop_right
            lower = height - self.crop_bottom
            return image.crop((left, upper, right, lower))

        elif isinstance(image, torch.Tensor):
            # Assumes tensor shape (C, H, W)
            _, h, w = image.shape
            top = self.crop_top
            bottom = h - self.crop_bottom
            left = self.crop_left
            right = w - self.crop_right
            return image[:, top:bottom, left:right]

        else:
            raise TypeError("Unsupported image type. Must be PIL.Image or torch.Tensor.")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CropSides":
        """
        Instantiates CropSides from configuration.

        Args:
            config (Dict): config dict with keys:
                - crop_left
                - crop_right
                - crop_top
                - crop_bottom

        Returns:
            CropSides instance.
        """
        return cls(
            crop_left=config["crop_left"],
            crop_right=config["crop_right"],
            crop_top=config["crop_top"],
            crop_bottom=config["crop_bottom"]
        )