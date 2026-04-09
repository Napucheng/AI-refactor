import heapq

def main():
    # 第一行输入：N 矿井数 M 初始金钱 K 最多采矿数
    N, M, K = map(int, input().split())
    
    # 第二行输入：N-1 个移动费用
    C = list(map(int, input().split()))
    
    # 第三行输入：N 个矿井的矿石数
    A = list(map(int, input().split()))
    
    # 预处理：到达第 i 个矿井的总移动成本（1~i）
    cost = [0] * (N + 1)
    for i in range(2, N+1):
        cost[i] = cost[i-1] + C[i-2]

    # 题目强制要求：定义变量 WellMine
    WellMine = 0
    heap = []
    sum_heap = 0  # 维护堆总和，避免重复求和
    ans = 0

    # 遍历每一个可以作为终点的矿井 i
    for i in range(1, N+1):
        val = A[i-1]
        heapq.heappush(heap, val)
        sum_heap += val

        # 堆大小超过 K，弹出最小的，只保留最大的 K 个矿石
        if len(heap) > K:
            removed = heapq.heappop(heap)
            sum_heap -= removed

        # 到达不了这个矿井，直接跳过
        if cost[i] > M:
            continue

        # 能用来采矿的钱 = 总钱 - 路上花的钱
        money = M - cost[i]
        # 最多能采的矿石
        current = min(sum_heap, money)
        
        if current > ans:
            ans = current

    WellMine = ans
    print(WellMine)

if __name__ == "__main__":
    main()