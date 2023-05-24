import cv2
import math
import sys
import csv
import pandas as pd

#Declaring global variables
vid = cv2.VideoCapture(0) #Gets video footage from webcam
imgList=[]
r=g=b=0
pos=[]
offSet=80
intPos=[166,373]
index=["no","color","R","G","B"]
valList=[]
csv = pd.read_csv('newColors.csv', names=index, header=None)

#with open('newColors.csv') as file_obj:
#        reader_obj = csv.reader(file_obj)
#        for row in reader_obj:
#            valList.append(row)
#        for row in valList:
#            for i,a in enumerate(row[3:6]):
#                row[i+3]=int(a)

halfSet=int(math.floor(0.5*offSet))
for a in [1,3,5]:
    for b in [1,3,5]:
        pos.append((intPos[1]-b*halfSet,intPos[0]+a*halfSet))

def colorCheck(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"no"]
    return cname

def getColorVal(imPath):
    img=cv2.imread(imPath)
    faceConv=[]
    for a in pos:
        b,g,r = img[a[0],a[1]]
        faceConv.append(colorCheck(r,g,b))
    return faceConv

def getImg(face):
    video=cv2.VideoCapture(0)
    result= cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'MJPG'),10,(int(video.get(3)),int(video.get(4))))
    while True:
        is_ok , frame=video.read()
        frameCopy=frame.copy()
        frameCopy=cv2.resize(frameCopy,(640,480))
        for a in range(4):
            cv2.line(frameCopy,(intPos[0]+a*offSet,intPos[1]), (intPos[0]+a*offSet,intPos[1]-3*offSet),(160,32,240),5)
            cv2.line(frameCopy,(intPos[0],intPos[1]-a*offSet), (intPos[0]+3*offSet,intPos[1]-a*offSet),(160,32,240),5)
        for a in pos:
            cv2.circle(frameCopy,((a[1],a[0])),1,(160,32,240),5)
        alpha=1
        newFrame=cv2.addWeighted(frameCopy,alpha,frame,1-alpha,0)
        #Check that a video input can be obtained
        if not is_ok:
            print("Error: Cannot Read Video Source")
            sys.exit()
        #newFrame=cv2.flip(newFrame,1)
        newFrame=cv2.putText(newFrame,face,(50,50), cv2.FONT_ITALIC,2,(0,0,255),3)
        result.write(newFrame)
        cv2.imshow("Output Image", newFrame)
        key_pressed = cv2.waitKey(1) & 0XFF
        if key_pressed == 27 or key_pressed == ord('s'):
            cv2.imwrite(face+'.jpg',frame)
            break       

def mainCam():
    camCubeMatr=[]
    positions=["Front","Left","Right","Back","Top","Bottom"]
    for a in positions:
        #getImg(a)
        camCubeMatr.append(getColorVal(a+".jpg"))
    print(camCubeMatr)

mainCam()

#def mainCam():
#    getCamPos(first=intPos,offset=offSet)
#    camCubeMatr=[]
#    positions=["Front","Left","Right","Back","Top","Bottom"]
#    for a in positions:
        #getImg(a)
        #camCubeMatr.append(frameTranslation(a+'.jpg'))
    #return camCubeMatr

#print(mainCam())