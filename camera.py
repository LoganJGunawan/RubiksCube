import pandas as pd
import cv2

#Declaring global variables
imgList=[]
r=g=b=0
camCubeMatr=[[],[],[],[],[],[]]

#Reading csv file with pandas and giving names to each column
index=["no","color","hex","R","G","B"]
csv = pd.read_csv('newColors.csv', names=index, header=None)

#function to calculate minimum distance from all colors and get the most matching color
def getColorCode(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color"]
            cnum = csv.loc[i,"no"]
            print(cnum)
        else:
            print("No work")
    #print(cname)
    return cnum

#Function to get the exact RGB value of a section in an image
def colorCheck(xpos,ypos,img):
    global r,g,b
    b,g,r=img[xpos][ypos]
    #b,g,r=imgList[img][xpos,ypos]
    b=int(b)
    g=int(g)
    r=int(r)

colorCheck(600,1200,cv2.imread('0.jpg'))
print(getColorCode(r,g,b))

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
