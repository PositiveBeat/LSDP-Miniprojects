import cv2
import numpy as np

import rasterio as ras
from rasterio.plot import show
from rasterio.windows import Window

from count_pumpkins import CountPumpkins


#Gathering size without saving file
pumpkins_ortho = 'input/Pumpkin_Field_Clearer.tif'

with ras.open(pumpkins_ortho) as Dataset:
    Width = Dataset.width
    Height = Dataset.height
    

pumpkin_count = 0

print(int(Height/7), int(Width / 12))

for i in range(0, Height, int(Height/7)):
    for j in range(0, Width, int(Width / 12)):
        
        with ras.open(pumpkins_ortho) as Dataset:
            window_location = Window(j, i, Width/12, Height/7)
            
            img = Dataset.read(window=window_location)
            
            # loaded image has a different shape than opencv image 
            # (Remove alpha channel, RGB -> BGR conversion)
            temp = img.transpose(1, 2, 0)
            t2 = cv2.split(temp)
            img_cv = cv2.merge([t2[2], t2[1], t2[0]])
            
            # Call pumpkin detection function
            pumpkins = CountPumpkins(img_cv, debug=True)
            pumpkin_count += pumpkins.pumpkin_count


print("Number of detected pumpkins: %d" % pumpkin_count)

