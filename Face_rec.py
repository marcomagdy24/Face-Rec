import face_recognition as fr
import os
import cv2
import numpy as np
import time


DETECTION_DIR = "Detection"
MODEL = "hog"
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
REC_COLOR  = (255,0,0)
FONT_COLOR = (200,200,200)

known_faces = []
known_names = []


def get_incoded_faces (image, name) :
    encoding = fr.face_encodings(image)[0]
    #print(encoding)
    known_faces.append(encoding)
    known_names.append(name) 




def train ():
    print("[INFO] Processing Known Faces")

    for name in os.listdir(KNOWN_FACES_DIR) :
        for filename in os.listdir(os.path.join(KNOWN_FACES_DIR,name)) :
            image = fr.load_image_file(os.path.join(KNOWN_FACES_DIR,name,filename))
            try :
                get_incoded_faces(image, name)
            except :
                print("[ERROR] No Encoding In Picture {}".format(filename))



def test_using_pics ():
    print("[INFO] Processing Unknown Faces")
    OUTPUT = 0
    for filename in os.listdir(UNKNOWN_FACES_DIR) :
        file_image = os.path.join(UNKNOWN_FACES_DIR,filename)
        image = fr.load_image_file(file_image)
        #img = cv2.imread(os.path.join(UNKNOWN_FACES_DIR,filename))
        #height, width = img.shape[:2]
        #image = cv2.resize(image,(width//8,height//8))
        locations = fr.face_locations(image,model=MODEL)
        encodings = fr.face_encodings(image,locations)
        print("[INFO] Detected {} Faces in {} image".format(len(locations), filename.split(".")[0]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for face_encoding, face_location in zip(encodings,locations):
            results = fr.compare_faces(known_faces,face_encoding,tolerance=TOLERANCE)
            match = None
            if True in results :
                match = known_names[results.index(True)]
                print("Match Found {}".format(match))
                top, right, bottom, left = face_location
                top_left = (left, top)
                bottom_right = (right, bottom)
                cv2.rectangle(image,top_left,bottom_right,REC_COLOR,FRAME_THICKNESS)
                cv2.putText(image, match,(left+10, bottom+15),cv2.FONT_HERSHEY_COMPLEX,0.5,FONT_COLOR,FONT_THICKNESS)
        cv2.imwrite(os.path.join(DETECTION_DIR,str(OUTPUT)+".jpg"),image)
        OUTPUT += 1
#         cv2.imshow("Detected Unknown",image)
#         k = cv2.waitKey(0)
#         if k==27:    # Esc key to stop
#             cv2.destroyAllWindows()




def test_using_camera ():
    video = cv2.VideoCapture(0)
    print("[INFO] Processing Unknown Faces")
    while True:
        _ , image = video.read()
        
        locations = fr.face_locations(image,model=MODEL)
        encodings = fr.face_encodings(image,locations)
        #print("[INFO] Detected {} Faces in {} image".format(len(locations), filename.split(".")[0]))
        for face_encoding, face_location in zip(encodings,locations):
            results = fr.compare_faces(known_faces,face_encoding,tolerance=TOLERANCE)
            match = None
            if True in results :
                match = known_names[results.index(True)]
                print("Match Found {}".format(match))
                top, right, bottom, left = face_location
                top_left = (left, top)
                bottom_right = (right, bottom)
                cv2.rectangle(image,top_left,bottom_right,REC_COLOR,FRAME_THICKNESS)
                cv2.putText(image, match,(left+10, bottom+15),cv2.FONT_HERSHEY_COMPLEX,0.5,FONT_COLOR,FONT_THICKNESS)
        cv2.imshow("Detected Unknown",image)
        if cv2.waitKey(1)& 0xFF==27 :    # Esc key to stop
            cv2.destroyAllWindows()
            break




def main ():
    train()

    test_using_pics()
    #test_using_camera()




if __name__ == '__main__' :
    main()






