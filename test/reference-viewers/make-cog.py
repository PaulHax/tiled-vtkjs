import sys

import numpy as np
from rasterio.io import MemoryFile
from rasterio.shutil import copy
from rasterio.transform import from_origin


size = 3072
y, x = np.ogrid[:size, :size]
rgb = np.stack([
    100 + ((x * 31 + y * 17) % 142),
    70 + ((x // 3 * 17 + y * 23) % 164),
    90 + ((x * 11 + y // 4 * 77) % 150),
]).astype("uint8")
with MemoryFile() as mem:
    with mem.open(
        driver="GTiff", width=size, height=size, count=3, dtype="uint8",
        crs="EPSG:3857", transform=from_origin(-2 * size, 2 * size, 4, 4)
    ) as image:
        image.write(rgb)
    with mem.open() as image:
        copy(image, sys.argv[1], driver="COG", BLOCKSIZE=256, OVERVIEWS="IGNORE_EXISTING")
