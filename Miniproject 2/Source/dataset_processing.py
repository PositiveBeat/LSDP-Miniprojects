import cv2
import numpy as np

from CSVprocessor import CSVprocessor
from exportkml import kmlclass

def visualise_flight_path():
    CSV = CSVprocessor('../Dataset/DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv', 274)
    gps = CSV.extract_float_columns([12, 13, 15])

    gps = np.transpose(gps).tolist()

    # Map showing the drone track during the drone flight
    kml = kmlclass()
    kml.begin('output/drone_track.kml', 'Drone Track', 'Drone track during the drone flight', 0.7)
    kml.trksegbegin ('', '', 'red', 'absolute') 
    for row in gps:
        lat = row[0]
        lon = row[1]
        
        # Add to file
        kml.pt(float(lat), float(lon), 100)
            
    kml.trksegend()
    kml.end()
    
    
def save_every_50th_frames():
    
    cap = cv2.VideoCapture('../Dataset/DJI_0199.MOV')
    i = 0
 
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == False:
            break
        
        if (i % 50 == 0):
            cv2.imwrite('../Dataset/50th_frames/frame'+str(i)+'.jpg', frame)
        i += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("All done!")
    
    
    
if __name__ == '__main__':
    
    # visualise_flight_path()
    
    save_every_50th_frames()
    
    
    
    