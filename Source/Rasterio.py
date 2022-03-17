from queue import Full
from re import I
from turtle import width
import rasterio as ras
from rasterio.plot import show
from rasterio.windows import Window

X = 0
Y = 0
i = 0
j = 0
#Gathering size without saving file
pumpkins = 'Pumpkin_Field_Clearer.tif'

with ras.open(pumpkins) as Dataset:
    Width = Dataset.width
    Height = Dataset.height
print(Width, Height)
#Window(1, 1, Width, Height)

while i<12:
    while j<12:
        with ras.open(pumpkins) as Dataset:
          window_location = Window.from_slices(slice(X,(Height/12)),slice(Y,(Width/12)),Height,Width)
          img = Dataset.read(window=window_location)

          Y += Width
          j += 1
    X += Height
    i += 1
