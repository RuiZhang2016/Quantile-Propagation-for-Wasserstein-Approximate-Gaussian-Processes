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


def sol(W,w,t):
    total_t = 0
    w_stack = [(w[0],0,t[0])]
    for i in range(1,len(t)):
        while len(w_stack)>0 and sum([e[0] for e in w_stack])+w[i] > W:
            min_id = 0
            min_v = w_stack[0][1]+w_stack[0][2]
            for j in range(1,len(w_stack)):
                if min_v > w_stack[j][1]+w_stack[j][2]:
                    min_v = w_stack[j][1]+w_stack[j][2]
                    min_id = j
            total_t = min_v
            w_stack.pop(min_id)
        w_stack+=[(w[i],total_t,t[i])]
    total_t = max([e[1]+e[2] for e in w_stack])
    return total_t

if __name__ == '__main__':
    print(sol(2,[1,1,1,1],[2,1,2,2]))

