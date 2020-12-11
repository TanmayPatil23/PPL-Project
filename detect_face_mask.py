import cv2
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_mouth.xml')

# Adjust threshold value in range 80 to 105 based on light at your end.
black_n_white_threshold = 80

# User message
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30) #cordinates of the bottom left-corner of the text to appear -- (x-cor, y-cor)
mask_font_color = (0, 255, 0) 
no_mask_font_color = (0, 0, 255)
thickness = 2 # in pixels
font_scale = 1 # to scale up or down the base size of the text
weared_mask = "Thank You for wearing MASK" 
not_weared_mask = "Please wear MASK to defeat Corona"

# Read video
#defining a video capture object..
cap = cv2.VideoCapture(0) # reads the video from the first port of camera

while 1:
    # Capturing video frame by frame
    #read gives by default 640 x 480 
    ret, img = cap.read() # cap.read() returns a bool value stored in ret. 
    img = cv2.flip(img,1) # pos flag = Y - axis, neg flag = both axes and 0 flag = X - axis , flips 2D array and returns an image

    # Convert Image into gray_image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #returns image with given color code

    # Convert image in black and white
    (thresh, black_and_white) = cv2.threshold(gray_image, black_n_white_threshold, 255, cv2.THRESH_BINARY) # returns a tuple of values and stores a gray_image image in the given src
    #cv2.imshow('black_and_white', black_and_white) -- fir getting the gray_image displayed

    # detect face
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5) # (image, scale_fator, min_neighbours) -- scalefactor to reduce the size or scale down the image

    # Face prediction for black and white
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.3, 5)


    if(len(faces) == 0 and len(faces_bw) == 0):
        # (src, text, font, fontscale, colot, thickness, linetype )
        cv2.putText(img, "No face found...", org, font, font_scale, mask_font_color, thickness, cv2.LINE_AA)
    elif(len(faces) == 0 and len(faces_bw) == 1):
        # It has been observed that for white mask covering mouth, with gray_image image face prediction is not happening
        cv2.putText(img, weared_mask, org, font, font_scale, mask_font_color, thickness, cv2.LINE_AA)
    else:
        # Draw rectangle on face
        # Faces contains tuples of cordinates of rectangles where rectangle contains the detected objects
        for (x, y, w, h) in faces:
            #(src, start, end, color[BGR TUPLE], thickness)
            # if thickness == -1; it fills the rectangle with that color
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray_image[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]


            # Detect lips counters
            mouth_viewed = mouth_cascade.detectMultiScale(gray_image, 1.5, 5)

        # Face detected but Lips not detected which means person is wearing mask or has his mouth covered with something
        if(len(mouth_viewed) == 0):
            cv2.putText(img, weared_mask, org, font, font_scale, mask_font_color, thickness, cv2.LINE_AA)
        else:
            for (mx, my, mw, mh) in mouth_viewed:
                if(y < my < y + h):
                    # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and person is not waring mask
                    cv2.putText(img, not_weared_mask, org, font, font_scale, no_mask_font_color, thickness, cv2.LINE_AA)
                    #cv2.rectangle(img, (mx, my), (mx + mh , my + mh), (0, 0, 255), 3)
                    break
    # Show frame with results
    #(window_name, image_to_show)
    cv2.imshow('Mask Detection', img)
    # waitkey(time in ms) -- gui to process the image and wait for a hit on keyboard if nothing done then returns -1 else ASCII
    k = cv2.waitKey(200) & 0xff
    if k == 27: # if esc key is hit
        break
# Release video
cap.release() # to close the video files and capturing device invoked by VideoCapture() method.
cv2.destroyAllWindows() #destroys all the windows created by imshow() method during the executions