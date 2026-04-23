import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    n = int(data[ptr]); ptr += 1
    m = int(data[ptr]); ptr += 1
    s = int(data[ptr]); ptr += 1

    w = [0] * (n + 1)
    v = [0] * (n + 1)
    minw = 10**18
    maxw = 0

    for i in range(1, n+1):
        wi = int(data[ptr]); ptr += 1
        vi = int(data[ptr]); ptr += 1
        w[i] = wi
        v[i] = vi
        if wi < minw:
            minw = wi
        if wi > maxw:
            maxw = wi

    intervals = []
    for _ in range(m):
        a = int(data[ptr]); ptr += 1
        b = int(data[ptr]); ptr += 1
        intervals.append((a, b))

    # 预分配数组，不再每次新建
    cnt = [0] * (n + 1)
    val = [0] * (n + 1)

    def calc(W):
        # 原地更新前缀和
        for i in range(1, n+1):
            cnt[i] = cnt[i-1]
            val[i] = val[i-1]
            if w[i] >= W:
                cnt[i] += 1
                val[i] += v[i]
        res = 0
        for l, r in intervals:
            c = cnt[r] - cnt[l-1]
            va = val[r] - val[l-1]
            res += c * va
        return res

    ans = 10**18
    l = minw
    r = maxw
    while l <= r:
        mid = (l + r) // 2
        y = calc(mid)
        dif = abs(y - s)
        if dif < ans:
            ans = dif
        if y > s:
            l = mid + 1
        else:
            r = mid - 1
    print(ans)

if __name__ == '__main__':
    main()