import cv2
from extract_num import *
from prediction import *

img_path = 'D:\\duke\\project\\sudoku\\sudoku3.jpg'
img = cv2.imread(img_path)
extract(img_path)
cropped_path ='D:\\duke\\project\\sudoku\\cropped\\'
weights_path='D:\\duke\\project\\sudoku\\model1.h5'
board = predict(cropped_path,weights_path)
cv2.imshow("input",img)
cv2.waitKey()
cv2.destroyAllWindows()

board=board.tolist()

def solve(sudoku):
    find = find_empty(sudoku)
    if not find:
        return True
    else:
        row,col = find
    for i in range(1,10):
        if valid(sudoku,i,(row,col)):
            sudoku[row][col]=i
            if solve(sudoku):
                return True
            sudoku[row][col]=0
    return False


def print_board(sudoku):
    for i in range(len(sudoku[0])):
        if(i%3 == 0 and i!=0):
            print("- - - - - - - - - - - - - - - - ")
        for j in range(len(sudoku[1])):
            if(j%3 == 0 and j!=0):
                print("| ",end =" ")
            if (j==8):
                print(sudoku[i][j])
            else:
                print(str(sudoku[i][j]) + " ",end=" ")
def find_empty(sudoku):
    for i in range(len(sudoku[0])):
        for j in range(len(sudoku[1])):
            if(sudoku[i][j]==0):
                return (i,j)
    return None

def valid(sudoku,num,pos):
    #check row
    for i in range(len(sudoku[0])):
        if(sudoku[pos[0]][i]==num and pos[1]!=i):
            return False

    #check column 
    for i in range(len(sudoku[1])):
        if(sudoku[i][pos[1]]==num and pos[0]!=i):
            return False
    #check box
    box_x = pos[0]//3
    box_y = pos[1]//3
    for i in range(box_x*3,box_x*3 +3):
        for j in range(box_y*3,box_y*3 +3):
            if(sudoku[i][j]== num and (i,j) != pos):
                return False
    return True


print_board(board)
solve(board)
print("--------------------------------------------")
print("--------------------------------------------")
print_board(board)

