import cv2

CASC_PATH = './face_detect/haarcascade/haarcascade_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
#print(CASC_PATH)
def face_detect(image):

    faces = cascade_classifier.detectMultiScale(image, scaleFactor = 1.3, minNeighbors = 5)
    if not len(faces) > 0:
        print ("Can not detect face information")
        exit()
        return None
    else:
        return faces