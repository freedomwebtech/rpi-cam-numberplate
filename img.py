import cv2
from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
cpt = 0
maxFrames = 100
while cpt < maxFrames:
    im= picam2.capture_array()
    im=cv2.flip(im,-1)
    cv2.imshow("Camera", im)
    cv2.imwrite('/home/pi/Downloads/rpicam/images/img_%d.jpg' %cpt, im)
    time.sleep(0.01)
    cpt += 1
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()