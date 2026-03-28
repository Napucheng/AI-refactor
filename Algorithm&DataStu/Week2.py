import sys
input = sys.stdin.readline

N = int(input())

L = [0] * (N + 1)
R = [0] * (N + 1)
exist = [True] * (N + 1)

# 初始只有 1
for i in range(2, N + 1):
    k, p = map(int, input().split())

    if p == 0:
        # 插到 k 左边
        L[i] = L[k]
        R[i] = k

        if L[k] != 0:
            R[L[k]] = i
        L[k] = i
    else:
        # 插到 k 右边
        R[i] = R[k]
        L[i] = k

        if R[k] != 0:
            L[R[k]] = i
        R[k] = i

M = int(input())

for _ in range(M):
    x = int(input())
    if not exist[x]:
        continue

    exist[x] = False

    if L[x] != 0:
        R[L[x]] = R[x]
    if R[x] != 0:
        L[R[x]] = L[x]

# 找队头（最左边）
head = 1
while L[head] != 0:
    head = L[head]

# 输出
res = []
cur = head
while cur != 0:
    if exist[cur]:
        res.append(str(cur))
    cur = R[cur]

print(" ".join(res))