n = int(input())
a = [[0] * (n + 2) for _ in range(n + 2)]

# 读取方格数据
while True:
    x, y, v = map(int, input().split())
    if x == 0 and y == 0:
        break
    a[x][y] = v

# 四维DP数组
dp = [[[[0] * (n + 2) for _ in range(n + 2)]
       for __ in range(n + 2)] for ___ in range(n + 2)]

# 开始动态规划
for i1 in range(1, n + 1):
    for j1 in range(1, n + 1):
        for i2 in range(1, n + 1):
            # 计算第二个人的列坐标
            j2 = i1 + j1 - i2
            if j2 < 1 or j2 > n:
                continue
            
            # 四种来源取最大
            max_prev = max(
                dp[i1-1][j1][i2-1][j2],
                dp[i1-1][j1][i2][j2-1],
                dp[i1][j1-1][i2-1][j2],
                dp[i1][j1-1][i2][j2-1]
            )
            
            # 计算当前值
            if i1 == i2 and j1 == j2:
                dp[i1][j1][i2][j2] = max_prev + a[i1][j1]
            else:
                dp[i1][j1][i2][j2] = max_prev + a[i1][j1] + a[i2][j2]

print(dp[n][n][n][n])