import cv2

# test image
img_file = 'car.png'

# create opencv image
img = cv2.imread(img_file)

# convert to grayscale
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# our pre-trained car classifier
classifier_file = 'car.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars 
cars=car_tracker.detectMultiScale(grayscale)
print(cars)

# draw rectangle around cars 
for(x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 5)


# display the image with faces spotted
cv2.imshow('Self Driving Car App',img)
cv2.waitKey()
print("Made by Vipul Sinha")