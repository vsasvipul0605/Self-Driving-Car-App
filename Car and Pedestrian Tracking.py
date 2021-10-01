import cv2

# create opencv image
img = cv2.imread('car.png')
# video = cv2.VideoCapture('Teslas Avoiding Accidents Compilation.mp4')
video = cv2.VideoCapture('videoplayback.mp4')

# convert to grayscale
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier('car.xml')
# haarcascade_fullbody classifier
pedestrian_tracker = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

while True:
    # read the current frame
    (read_successful, frame) = video.read()

    # safe coding
    if read_successful:
        # convert to grayscale 
        grayscale_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars 
    cars = car_tracker.detectMultiScale(grayscale_vid)
    # detect pedestrians 
    pedestrians = pedestrian_tracker.detectMultiScale(grayscale_vid)

    # draw rectangle around cars 
    for(x, y, w, h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
    # draw rectangle around pedestrians 
    for(x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 5)

    # display the video
    cv2.imshow('Self Driving Car App',frame)
    key = cv2.waitKey(1)

    # stop if q key pressed
    if key== ord('q'): # ord() return unicode of passing character
        break   

# release the VideoCapture object
video.release()

"""
# detect cars 
cars=car_tracker.detectMultiScale(grayscale)
print(cars)

# draw rectangle around cars 
for(x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 5)


# display the image with faces spotted
cv2.imshow('Self Driving Car App',img)
cv2.waitKey()
"""
print("Made by Vipul Sinha")