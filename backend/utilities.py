import cv2 as cv
import numpy as np
from ultralytics import YOLO
model = YOLO('yolo11n.pt')

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    #Resize frame to a set dimension
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Read Image
#img = cv.imread('C:\Users\Conor\Videos\ConcussionAssessment\Concussion Hits\CJStroudConcussion.mp4')
# Show Image
#Display Window
#cv.imshow("Display window", img)
#cv.waitKey(0)

#Read Video
capture = cv.VideoCapture('C:/Users/Conor/Videos/ConcussionAssessment/Concussion Hits/CJStroudConcussion.mp4')
#While loop to read video 
#Frame by Frame
while True:
    isTrue, frame = capture.read()
    results = model(frame)
    annotated_frame = results[0].plot()

    frameResized = rescaleFrame(annotated_frame )
    #Display every frame
    #cv.imshow("Video", frame)
    cv.imshow("Video Resized", frameResized)
    #Stop the video~ If d is pressed
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()



