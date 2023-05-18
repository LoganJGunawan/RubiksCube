import random
import numpy as np

moveList=[]
solvedMatrix=[]
testCubeMatrix=[[0,1,2,3,4,5,6,7,8],[1,1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2,2],[3,3,3,3,3,3,3,3,3],[4,4,4,4,4,4,4,4,4],[5,5,5,5,5,5,5,5,5]]
def initMatr():
    global cubeMatrix, solvedMatrix
    tempM=[]
    for a in range(6):
        for i in range(9):
            tempM.append(a)
    solvedMatrix=cubeMatrix=np.reshape(np.array(tempM),(6,9))

initMatr()

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

testTurn=[[0,1,2,3,4,5,6,7,8]]

def turn(cubeMatr,face,clockwise):           #Currently Working (10/03/2023)
    tempHolding=[]
    for a in cubeMatr[face]:
        tempHolding.append(a)
    antClockFacePos=[6,3,0,7,4,1,8,5,2]
    clockFacePos=[2,5,8,1,4,7,0,3,6]
    if clockwise:
        for a in range(len(clockFacePos)):
            cubeMatr[face][a]=tempHolding[clockFacePos[a]]
    else:
        for a in range(len(antClockFacePos)):
            cubeMatr[face][a]=tempHolding[antClockFacePos[a]]
    return cubeMatr

print(turn(testTurn,0,False))

def turnOF(face,turnB):
    global cubeMatrix
    moveList.append([face,turnB])
    fTurnArr=[]
    if face==0:                 #TESTED SUCCESFUL 17/05
        fTurnArr=[[[0,1,3,2],[[6,7,8],[6,7,8],[6,7,8],[6,7,8]],[2,0,1,3],[5,True]],    #Bottom Left Done
                 [[0,1,3,2],[[6,7,8],[6,7,8],[6,7,8],[6,7,8]],[1,3,2,0],[5,False]],    #Bottom Right Done
                 [[0,4,3,5],[[2,5,8],[2,5,8],[6,3,0],[2,5,8]],[5,0,4,3],[2,True]],    #Right Top Done
                 [[0,4,3,5],[[2,5,8],[2,5,8],[6,3,0],[2,5,8]],[4,3,5,0],[2,False]],    #Right Bottom Done
                 [[0,4,3,5],[[0,3,6],[0,3,6],[8,5,2],[0,3,6]],[5,0,4,3],[1,True]],    #Left Top Done
                 [[0,4,3,5],[[0,3,6],[0,3,6],[8,5,2],[0,3,6]],[4,3,5,0],[1,False]],    #Left Bottom Done
                 ]   
    elif face==1:              #TESTED SUCCESFUL 17/05
        fTurnArr=[[[1,3,2,0],[[6,7,8],[6,7,8],[6,7,8],[6,7,8]],[0,1,3,2],[5,True]],    #Bottom Left Done
                 [[1,3,2,0],[[6,7,8],[6,7,8],[6,7,8],[6,7,8]],[3,2,0,1],[5,False]],    #Bottom Right Done
                 [[1,4,2,5],[[2,5,8],[8,7,6],[6,3,0],[0,1,2]],[5,1,4,2],[0,True]],    #Right Top Done
                 [[1,4,2,5],[[2,5,8],[8,7,6],[6,3,0],[0,1,2]],[4,2,5,1],[0,False]],    #Right Bottom Done
                 [[1,4,2,5],[[0,3,6],[2,1,0],[8,5,2],[6,7,8]],[5,1,4,2],[3,True]],    #Left Top Done
                 [[1,4,2,5],[[0,3,6],[2,1,0],[8,5,2],[6,7,8]],[4,2,5,1],[3,False]],    #Left Bottom Done
                 ]   
    elif face==2:              #TESTED SUCCESFUL 17/05
        fTurnArr=[[[2,0,1,3],[[6,7,8],[6,7,8],[6,7,8],[6,7,8]],[3,2,0,1],[5,True]],    #Bottom Left Done
                 [[2,0,1,3],[[6,7,8],[6,7,8],[6,7,8],[6,7,8]],[0,1,3,2],[5,False]],    #Bottom Right Done
                 [[2,4,1,5],[[2,5,8],[0,1,2],[5,3,0],[8,7,6]],[5,2,4,1],[3,True]],    #Right Top Done
                 [[2,4,1,5],[[2,5,8],[0,1,2],[5,3,0],[8,7,6]],[4,1,5,2],[3,False]],    #Right Bottom Done
                 [[2,4,1,5],[[0,3,6],[6,7,8],[8,5,2],[2,1,0]],[5,2,4,1],[0,True]],    #Left Top Done
                 [[2,4,1,5],[[0,3,6],[6,7,8],[8,5,2],[2,1,0]],[4,1,5,2],[0,False]]     #Left Bottom Done
                 ]   
    elif face==3:              #Tested Succesful
        fTurnArr=[[[3,2,0,1],[[6,7,8],[6,7,8],[6,7,8],[6,7,8]],[1,3,2,0],[5,True]],    #Bottom Left Done
                 [[3,2,0,1],[[6,7,8],[6,7,8],[6,7,8],[6,7,8]],[2,0,1,3],[5,False]],    #Bottom Right Done
                 [[3,5,0,4],[[2,5,8],[6,3,0],[6,3,0],[6,3,0]],[5,0,4,3],[1,True]],    #Right Top Done
                 [[3,5,0,4],[[2,5,8],[6,3,0],[6,3,0],[6,3,0]],[4,3,5,0],[1,False]],    #Right Bottom Done
                 [[3,5,0,4],[[0,3,6],[8,5,2],[8,5,2],[8,5,2]],[5,0,4,3],[2,True]],    #Left Top Done
                 [[3,5,0,4],[[0,3,6],[8,5,2],[8,5,2],[8,5,2]],[4,3,5,0],[2,False]],    #Left Bottom Done
                 ]
    elif face==4:             #Tested Succesful
        fTurnArr=[[[4,1,5,2],[[6,7,8],[8,5,2],[2,1,0],[0,3,6]],[2,4,1,5],[0,True]],    #Bottom Left Done
                 [[4,1,5,2],[[6,7,8],[8,5,2],[2,1,0],[0,3,6]],[1,5,2,4],[0,False]],    #Bottom Right Done
                 [[4,3,5,0],[[2,5,8],[6,3,0],[2,5,8],[2,5,8]],[0,4,3,5],[2,True]],     #Right Top Done    
                 [[4,3,5,0],[[2,5,8],[6,3,0],[2,5,8],[2,5,8]],[3,5,0,4],[2,False]],    #Right Bottom Done
                 [[4,3,5,0],[[0,3,6],[8,5,2],[0,3,6],[0,3,6]],[0,4,3,5],[1,True]],     #Left Top Done    
                 [[4,3,5,0],[[0,3,6],[8,5,2],[0,3,6],[0,3,6]],[3,5,0,4],[1,False]],    #Left Bottom Done
                 ]
    elif face==5:             #Tested Succesful
        fTurnArr=[[[5,1,4,2],[[6,7,8],[0,3,6],[2,1,0],[8,5,2]],[2,5,1,4],[3,True]],    #Bottom Left Done    
                 [[5,1,4,2],[[6,7,8],[0,3,6],[2,1,0],[8,5,2]],[1,5,2,5],[3,False]],    #Bottom Right Done    
                 [[5,0,4,3],[[2,5,8],[2,5,8],[2,5,8],[6,3,0]],[3,5,0,4],[2,True]],     #Right Top Done
                 [[5,0,4,3],[[2,5,8],[2,5,8],[2,5,8],[6,3,0]],[0,4,3,5],[2,False]],    #Right Bottom Done
                 [[5,0,4,3],[[0,3,6],[0,3,6],[0,3,6],[8,5,2]],[3,5,0,4],[1,True]],     #Left Top Done
                 [[5,0,4,3],[[0,3,6],[0,3,6],[0,3,6],[8,5,2]],[0,4,3,5],[1,False]],    #Left Bottom Done
                 ]
    reworkedSwap(fTurnArr[turnB][0],fTurnArr[turnB][1],fTurnArr[turnB][2],cubeMatrix)
    turn(cubeMatrix,fTurnArr[turnB][3][0],fTurnArr[3][1])

def checkBasicPossibility():
    basicPoss=[0,0,0,0,0,0]
    for a in cubeMatrix:
        for b in a:           
            basicPoss[b-1]+=1
    for a in basicPoss:
        if a!=9:
            print("Basic Check 1 Failed")

def shuffle():
    for i in range(random.randint(1,99)):
        turnOF(random.randint(0,3),random.randint(0,5))
        checkBasicPossibility()
    print(cubeMatrix)

def Eval():
    print("Moves to Completion: "+str(len(moveList)))
    print("Move List: "+moveList)
                

def testingProgram():
    global cubeMatrix
    print("What would you like to do:\n END = End program\n R = Reset Cube\n Face,Turn = Move Face,Turn\n Shuff = Shuffle")
    MInp=input("What would you Like to do: \n")
    while MInp!="END":
        if MInp=="R":
            initMatr()
            print("Cube has been reset")
        elif MInp=="Shuff":
            shuffle()
        else:
            try:
                print("Face: "+MInp[0]+" Turn: "+MInp[1])
                turnOF(int(MInp[0]),int(MInp[1]))
                print(cubeMatrix)
            except:
                print("Command Not Reconized Try Again")
        MInp=input("What would you Like to do: \n")
    print("Ending Program")

#testingProgram()