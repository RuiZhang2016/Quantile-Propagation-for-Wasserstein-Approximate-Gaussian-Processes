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

if __name__ == '__main__':
    a = [1,2,3,4]
    lrks = [[1,4,2],[1,3,2],[1,2,1]]
    sol(a,lrks)

