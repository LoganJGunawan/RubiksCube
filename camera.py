import pandas as pd
import cv2
from function import colorCheck, getColorCode
import keyboard

#Declaring global variables
vid = cv2.videoCapture(0) #Gets video footage from webcam
imgList=[]
r=g=b=0
camCubeMatr=[[],[],[],[],[],[]]

#Reading csv file with pandas and giving names to each column
index=["no","color","hex","R","G","B"]
csv = pd.read_csv('newColors.csv', names=index, header=None)

colorCheck(600,1200,cv2.imread('0.jpg'))
print(getColorCode(r,g,b))

def camMain(vid):
    frames=[]
    faces=["Front","Left","Right","Back","Top","Bottom"]
    for a in len(faces):
        print("Show "+faces[a]+" face\n")
        ret,frame = vid.read()
        cv2.imshow('Video',frame)
        cv2.waitKey(0)
        keyboard.wait('space')
        frames.append(frame)
    vid.release()
    cv2.destroyAllWindows()
    return frames

def showImg(list):
    for a in list:
        keyboard.wait('space')
        cv2.imshow('Img',cv2.imread(a))
        cv2.waitKey(0)
    cv2.destroyAllWindows()

imgList=camMain(vid=vid)
showImg(imgList)

#Loading all images
#try:
#    for i in range(0):
#        imgList.append(cv2.imread(str(i)+'.jpg'))
#        print("Pass 1: Image "+str(i)+" loaded.")
#except:
#    print("Error 1: Images Not Loaded")

#Creating the matrix representation of initial cube
#def getMatrix():
#    matrix=[]
#    for a in range(6):
#        matrix.append([])
#        for b in range(8):
#            matrix[a].append(getColorCode(r,g,b))
#    return matrix
