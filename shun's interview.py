# def sol(doors, balls):
#     score = 0
#     door_id = 0
#     nd = len(doors)
#     for b in balls:
#         while door_id < nd:
#             if doors[door_id][0]<=b<=doors[door_id][1]:
#                 score += 1
#                 break
#             elif b<doors[door_id][0]:
#                 break
#             else:
#                 door_id +=1
#     return score
#
# def main():
#     n, m = [int(e) for e in input().split(' ')]
#     doors = [[int(e) for e in input().split(' ')] for _ in range(n)]
#     doors = sorted(doors)
#     balls = [int(input()) for _ in range(m)]
#     balls = sorted(balls)
#     score = sol(doors,balls)
#     print(score)

from queue import PriorityQueue as PQ
# def sol(a,lrks):
#     etn = [0]*len(a) # binary
#     for itvl in lrks:
#         l,r,k = itvl
#         l,r = l-1,r-1
#         end = -2
#         if r-l+1 <= k:
#             for i in range(l,r+1):
#                 if etn[i] == 0:
#                     end = i
#                     etn[end] = 1
#         else:
#             s = sorted(a[l:r+1])
#             i,j = 0,0
#             while i <k and j<r-l+1:
#                 if etn[j] == 0:
#                     end =a.index(s[j])
#                     etn[end] = 1
#                     i+= 1
#                 j+=1
#         print(end+1)




def my_sort(a):
    from operator import itemgetter
    a = sorted(a,key=itemgetter(0,2,1))
    return a

if __name__ == '__main__':
    a = [['Applpe',1,80],['Apple',2,62],['Apple',5,62],['Apple',4,73],['Orange',4,65],['Orange',1,90],['Apple',3,91]]
    print(my_sort(a))



def getValidMoves(board, tile):
    validMoves = []
    for x in range(8):
        for y in range(8):
            if isValidMove(board, tile, x, y) != False:validMoves.append([x, y]);
    return validMoves

def isOnCorner(x, y):
    return (x == 0 and y == 0) or (x == 7 and y == 0) or (x == 0 and y == 7) or (x == 7 and y == 7)


def isValidMove(board, tile, xstart, ystart):
    if not isOnBoard(xstart, ystart) or board[xstart][ystart] != 'none': return False;
    board[xstart][ystart] = tile
    otherTile = 'white' if tile == 'black' else 'black'
    tilesToFlip = []
    for xdirection, ydirection in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
        x, y = xstart, ystart
        x, y = x + xdirection, y + ydirection
        if isOnBoard(x, y) and board[x][y] == otherTile:
            x, y = x + xdirection, y + ydirection
            if not isOnBoard(x, y):
                continue
            while board[x][y] == otherTile:
                x, y = x + xdirection, y + ydirection
                if not isOnBoard(x, y):break;

            if not isOnBoard(x, y):continue;
            if board[x][y] == tile:
                while True:
                    x,y = x-xdirection,y-ydirection
                    if x == xstart and y == ystart:break;
                    tilesToFlip.append([x, y])

    board[xstart][ystart] = 'none'
    if len(tilesToFlip) == 0:  return False;
    return tilesToFlip

def isOnBoard(x, y):
    return x >= 0 and x <= 7 and y >= 0 and y <= 7

def makeMove(board, tile, xstart, ystart):
    tilesToFlip = isValidMove(board, tile, xstart, ystart)

    if tilesToFlip == False:return False;

    board[xstart][ystart] = tile
    for x, y in tilesToFlip:board[x][y] = tile;
    return True

def getScoreOfBoard(board):
    xscore = 0
    oscore = 0
    for x in range(8):
        for y in range(8):
            if board[x][y] == 'black':
                xscore += 1
            if board[x][y] == 'white':
                oscore += 1
    return {'black': xscore, 'white': oscore}
