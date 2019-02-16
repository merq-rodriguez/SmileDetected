import cv2
import numpy as np
import sys
import uuid

class Smile:
    def __init__(self, facePath, smilePath):
        self.faceCascade = cv2.CascadeClassifier(facePath) 
        self.smileCascade = cv2.CascadeClassifier(smilePath)
        self.sF = 1.05
    #Iniciar captura de video
    def start(self):
        cap = cv2.VideoCapture(0)
        cap.set(3,640)
        cap.set(4,480)
        self.runWebcam(cap)

    # CascadeClassifier :: detectMultiScale
    # Detecta objetos de diferentes tamaños en la imagen de entrada. 
    # Los objetos detectados se devuelven como una lista de rectángulos.
    def getFaceDetectMultiScale(self, gray):
        return self.faceCascade.detectMultiScale(
            gray,
            scaleFactor= self.sF,
            minNeighbors=8,
            minSize=(55, 55),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
    def getSmileDetectMultiScale(self, roi_gray):
        return  self.smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.7,
            minNeighbors=22,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
    )

    def capturePhoto(self, frame):
        photo = str(uuid.uuid1()) +".png"
        cv2.imwrite(photo, frame)
        print("Foto tomada correctamente")
    

    def runWebcam(self, cap):
        isSmile = False
        while (cap.isOpened() and isSmile == False):
            ret, frame = cap.read() # Captura cuadro por cuadro
            img = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            faces = self.getFaceDetectMultiScale(gray)

            # Dibuja un rectángulo alrededor de las caras.
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                smile = self.getSmileDetectMultiScale(roi_color)

                for (x, y, w, h) in smile:
                    #print("Found "+ str(len(smile))+ " smiles!")
                    print("Hemos detectado una sonrisa :D!")
                    isSmile = True
                    self.capturePhoto(frame) #Capura foto
                cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)

            cv2.imshow('Smile Detector', frame)
            c = cv2.waitKey(7) % 0x100
            if c == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Terminó")
    

    


if __name__ == "__main__":
    smi = Smile("haarcascade_frontalface_default.xml", "haarcascade_smile.xml")
    smi.start()
