def turnOF(face,turnB):
    global cubeMatrix
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

def shuffle():
    for i in range(random.randint(1,99)):
        turnOF(random.randint(0,3),random.randint(0,5))
    return cubeMatrix