File Explanations:
function.py   - Contains most major functions and test functions
camera.py     - Contains code to process images and return initial cube State
learning.py   - Algorithm to study cube solving
solve.py      - Code to return information to Arduino
test.py       - Miscellaneous testing
newColors.csv - CSV file used to color match and identify colors in a cube by camera.py
learn.csv     - CSV file used to learn cube solving by learning.py

Representation:
Faces:       Turn:                               Color:
0 - Front    Bottom turned Left                  Yellow
1 - Left     Bottom turned Right                 Blue
2 - Right    Right turned Up                     Green
3 - Back     Right turned Down                   White
4 - Top      Left turned Up                      Orange
5 - Bottom   Left turned Down                    Red
Examples:
00 - The front face's bottom side turned Left       [0,2,3       [Yellow, Green, White
32 - Back face's right side turned up                5,5,1        Red, Red, Blue
20 - Right face's bottom side turned Left            4,2,0]       Orange, Green, Yellow]

Unused Move:
X6 & X7: 6 and 7 would represent the move of turning the top side of the face of a cube Left and Right respectively
         however, faces 0-3 are unable to use this move due to the lack of a motor on the top face to turn it. while
         technically faces 4 and 5 are able to perform this move due to having a motor on all 4 sides of its face,
         there is minimal need to implement this as currently (19/05/2023).

Log:
11/05/2023 function.py is completed. Will test soon
15/05/2023 Basic testing of function.py finished
15/05/2023 Bugs fixed in function.py
19/05/2023 Optimization in function.py: newTurn()