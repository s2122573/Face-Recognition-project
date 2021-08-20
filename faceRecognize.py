import cv2
import numpy as np


class FaceRecognize():
    """
    人脸识别类
    """
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                                + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                                + "haarcascade_eye.xml")
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                                + "haarcascade_smile.xml")

    """
    输入图片路径，返回截取的人脸图像
    input:
        filename 输入图片路径
        flag 输出图像标志，如果为True，输出为灰度，否则为彩色
    return:
        检测到人脸，则为人脸区域的图像，否则为None
     """
    def getFace(self, filename, flag=True):
        img = cv2.imread(filename)
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray_img,
                                            scaleFactor=1.05,
                                            minNeighbors=5,
                                            minSize=(50, 50),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) > 0:
            # print(faces)
            (x, y, w, h) = faces[0]
            return gray_img[y:y+h, x:x+w]
        else:
            return None


    # 画上检测的人脸
    def showFacesRects(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_img,
                                            scaleFactor=1.05,
                                            minNeighbors=5,
                                            minSize=(32, 32),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
        max = -1
        mX, mY, mW, mH = 0, 0, 0, 0
        for (x, y, w, h) in faces:
            if w*h > max:
                max = w*h
                mX, mY, mW, mH = x, y, w, h
        
        # cv2.rectangle(img, (mX, mY), (mX+mW, mY+mH), (255, 0, 0), 1)
        # cv2.circle(img, (int(mX+mW/2), int(mY+mH/2)), 3, (0, 0, 255))
        
        if max == -1:
            return None
        return gray_img[y:y+h, x:x+w]
    
    # 检测测试函数
    def showDetected(self, filename):
        face_cascade = self.face_cascade
        eye_cascade = self.eye_cascade
        smile_cascade = self.smile_cascade
        img = cv2.imread(filename)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_img,
                                            scaleFactor=1.05,
                                            minNeighbors=5,
                                            minSize=(32, 32),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:

            faceROI = gray_img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(faceROI, 
                                                scaleFactor=1.1, 
                                                minNeighbors=10, 
                                                minSize=(15, 15), 
                                                flags=cv2.CASCADE_SCALE_IMAGE)

            smiles = smile_cascade.detectMultiScale(faceROI,
                                                    scaleFactor=1.1, 
                                                    minNeighbors=10,
                                                    minSize=(15, 15),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
            
            for (eX, eY, eW, eH) in eyes:
                ptA = (x+eX, y+eY)
                ptB = (x+eX+eW, y+eY+eH)
                cv2.rectangle(img, ptA, ptB, (0, 255, 0), 2)
            
            for (sX, sY, sW, sH) in smiles:
                ptA = (x+sX, y+sY)
                ptB = (x+sX+sW, y+sY+sH)
                cv2.rectangle(img, ptA, ptB, (0, 0, 255), 2)

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)

        #     cv2.namedWindow("Face")
        #     cv2.imshow("Face", faceROI)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # cv2.namedWindow("Face Detected!")
        # cv2.imshow("Face Detected", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def showImg(img):
    cv2.namedWindow("show Image!")
    cv2.imshow("show image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recogonizer = FaceRecognize()
    face = recogonizer.getFace("datasets\CASME_A\Section A\sub01\EP01_5\EP01_5-4.jpg")
    showImg(face)
    # recognizer.staticDetect("datasets\CASME_A\Section A\sub01\EP01_5\EP01_5-1.jpg")




