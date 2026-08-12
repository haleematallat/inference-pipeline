import pytest
from PIL import Image


@pytest.fixture
def solid_image():
    def create(color: tuple[int, int, int]) -> Image.Image:
        return Image.new("RGB", (8, 8), color)

    return create
