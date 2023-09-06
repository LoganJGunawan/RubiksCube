import random
import numpy as np
import math

#CUBE FUNCTIONS
#CUBE FUNCTIONS

def resetCube():
    tempM=[]
    for a in range(6):
        for i in range(9):
            tempM.append(a)
    cubeMatrix=np.reshape(np.array(tempM),(6,9))
    return cubeMatrix

solvedCubeMatrix=resetCube()

def swap(side1,side2,side3,cubeMatr):        #BROKEN AND USELESS (10/05/2023)    REPLACED(11/05/2023)
    tempHolding=[[],[],[],[]]
    for a in range(len(side1)):
        for i in side2:
            tempHolding[a].append(cubeMatr[side1[a]][i])
    for b in range(len(side3)):
        for a in range(len(side2)):
            cubeMatr[side3[b]][side2[a]]=tempHolding[b][a]
    return cubeMatr

def reworkedSwap(faceFrom,sideAffectList,faceTo,cubeMatr):         #Working tested no bugs so far (10/05/2023)
    tempHolding=[[],[],[],[]]
    c=0
    for a in range(4):
        holding=[]
        for b in cubeMatr[faceFrom[a]]:
            holding.append(b)
        for e in sideAffectList[a]:
            tempHolding[a].append(holding[e])
    for a in range(len(faceTo)):
        d=0
        faceFromTo=faceFrom.index(faceTo[a])
        for b in range(3):
            cubeMatr[faceTo[a]][sideAffectList[faceFromTo][b]]=tempHolding[c][d]
            d+=1
        c+=1
    newCubeMatr=cubeMatr
    return newCubeMatr

def turn(cubeMatr,face,clock):           #Currently Working (10/03/2023)
    tempHolding=[]
    for a in cubeMatr[face]:
        tempHolding.append(a)
    antClockFacePos=[6,3,0,7,4,1,8,5,2]
    clockFacePos=[2,5,8,1,4,7,0,3,6]
    if clock:
        for a in range(len(clockFacePos)):
            cubeMatr[face][a]=tempHolding[clockFacePos[a]]
    else:
        for a in range(len(antClockFacePos)):
            cubeMatr[face][a]=tempHolding[antClockFacePos[a]]
    return cubeMatr

#More optimized turnOF
def oldTurn(cubematrix,face,turnType):
    newCubeMatrix=cubematrix
    dupeList=["12,24,40","13,25,41","04,32,44,54","05,33,45,55","02,34,42,52","03,35,42,52","14,22,50","15,23,51","00,10,20,30","01,11,21,31"]
    moveList=[
              [[1,4,2,5],[[2,5,8],[8,7,6],[6,3,0],[0,1,2]],[4,2,5,1]],    
              [[1,4,2,5],[[2,5,8],[8,7,6],[6,3,0],[0,1,2]],[5,1,4,2]],
              [[0,4,3,5],[[0,3,6],[0,3,6],[8,5,2],[0,3,6]],[4,3,5,0]],    
              [[0,4,3,5],[[0,3,6],[0,3,6],[8,5,2],[0,3,6]],[5,0,4,3]],
              [[0,4,3,5],[[2,5,8],[2,5,8],[6,3,0],[2,5,8]],[4,3,5,0]],    
              [[0,4,3,5],[[2,5,8],[2,5,8],[6,3,0],[2,5,8]],[5,0,4,3]],
              [[1,4,2,5],[[0,3,6],[2,1,0],[8,5,2],[6,7,8]],[5,1,4,2]],
              [[1,4,2,5],[[0,3,6],[2,1,0],[8,5,2],[6,7,8]],[4,2,5,1]],
              [[0,1,3,2],[[6,7,8],[6,7,8],[6,7,8],[6,7,8]],[2,0,1,3]],
              [[0,1,3,2],[[6,7,8],[6,7,8],[6,7,8],[6,7,8]],[1,3,2,0]]
    ]   
    for a in dupeList:
        if str(face)+str(turnType) in a:
            if dupeList.index(a)%2==0:
                TF=True
            else:
                TF=False
            move=moveList[dupeList.index(a)]
            newCubeMatrix=reworkedSwap(move[0],move[1],move[2],newCubeMatrix)
            if math.floor(dupeList.index(a)/2)==4:
                newCubeMatrix=turn(newCubeMatrix,5,TF)
            else:    
                newCubeMatrix=turn(newCubeMatrix,math.floor(dupeList.index(a)/2),TF)
    return newCubeMatrix

def newTurn(cubematrix,action):
    newCubeMatrix=cubematrix
    moveList=[
              [[1,4,2,5],[[2,5,8],[8,7,6],[6,3,0],[0,1,2]],[4,2,5,1]],    
              [[1,4,2,5],[[2,5,8],[8,7,6],[6,3,0],[0,1,2]],[5,1,4,2]],
              [[0,4,3,5],[[0,3,6],[0,3,6],[8,5,2],[0,3,6]],[4,3,5,0]],    
              [[0,4,3,5],[[0,3,6],[0,3,6],[8,5,2],[0,3,6]],[5,0,4,3]],
              [[0,4,3,5],[[2,5,8],[2,5,8],[6,3,0],[2,5,8]],[4,3,5,0]],    
              [[0,4,3,5],[[2,5,8],[2,5,8],[6,3,0],[2,5,8]],[5,0,4,3]],
              [[1,4,2,5],[[0,3,6],[2,1,0],[8,5,2],[6,7,8]],[5,1,4,2]],
              [[1,4,2,5],[[0,3,6],[2,1,0],[8,5,2],[6,7,8]],[4,2,5,1]],
              [[0,1,3,2],[[6,7,8],[6,7,8],[6,7,8],[6,7,8]],[2,0,1,3]],
              [[0,1,3,2],[[6,7,8],[6,7,8],[6,7,8],[6,7,8]],[1,3,2,0]]
    ]   
    if action%2==0:
        TF=True
    else:
        TF=False
    move=moveList[action]
    newCubeMatrix=reworkedSwap(move[0],move[1],move[2],newCubeMatrix)
    if math.floor(action/2)==4:
        newCubeMatrix=turn(newCubeMatrix,5,TF)
    else:    
        newCubeMatrix=turn(newCubeMatrix,math.floor(action/2),TF)
    return newCubeMatrix

def checkSolved(matr):
    for index,a in enumerate(matr):
        zipped=zip(a,solvedCubeMatrix[index])
        for a in zipped:
            #print(a)
            if a[0]==a[1]:
                continue
            else:
                return False
    return True

def shuffle():
    cubeMatrix=resetCube()
    for i in range(15):
        a=random.randint(0,9)
        newTurn(cubeMatrix,a)
    return cubeMatrix

def checkAlign(state):
    high=0
    for a in state:
        if np.count_nonzero(a == a[0]) == 9:
            high+=1
    return high

def testingProgram():
    global cubeMatrix
    print("What would you like to do:\n END = End program\n R = Reset Cube\n Face,Turn = Move Face,Turn\n Shuff = Shuffle")
    MInp=input("What would you Like to do: \n")
    while MInp!="END":
        if MInp=="R":
            print("Cube has been reset")
        elif MInp=="Shuff":
            shuffle()
        else:
            try:
                print("Face: "+MInp[0]+" Turn: "+MInp[1])
                newTurn(int(MInp[0]),int(MInp[1]))
                print(cubeMatrix)
            except:
                print("Command Not Reconized Try Again")
        MInp=input("What would you Like to do: \n")
    print("Ending Program")

#testingProgram()

if __name__ == "__main__":
    cube = resetCube()
    #cube = newTurn(cube,0)
    print(cube)
    print(checkAlign(cube))