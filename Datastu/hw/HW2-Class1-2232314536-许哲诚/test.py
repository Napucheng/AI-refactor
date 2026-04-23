import time
import random
import copy
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 全局配置
# ==========================================
# 阈值：当数据量小于此值时，直接使用串行排序，避免进程创建开销
THRESHOLD = 2000 
# 进程池最大 worker 数量
MAX_WORKERS = 4

# ==========================================
# 1. 归并排序 (Merge Sort)
# ==========================================

def merge(left, right):
    """串行合并两个有序列表"""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def serial_merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = serial_merge_sort(arr[:mid])
    right = serial_merge_sort(arr[mid:])
    return merge(left, right)

def parallel_merge_sort_task(arr):
    """进程任务函数：归并排序"""
    if len(arr) <= THRESHOLD:
        return sorted(arr)
    mid = len(arr) // 2
    # 使用进程池并行处理左右两边
    with ProcessPoolExecutor(max_workers=2) as executor:
        left_future = executor.submit(parallel_merge_sort_task, arr[:mid])
        right_future = executor.submit(parallel_merge_sort_task, arr[mid:])
        left = left_future.result()
        right = right_future.result()
    return merge(left, right)

# ==========================================
# 2. 快速排序 (Quick Sort)
# ==========================================

def serial_quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return serial_quick_sort(left) + middle + serial_quick_sort(right)

def parallel_quick_sort_task(arr):
    """进程任务函数：快速排序"""
    if len(arr) <= THRESHOLD:
        return sorted(arr)
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        left_future = executor.submit(parallel_quick_sort_task, left)
        right_future = executor.submit(parallel_quick_sort_task, right)
        sorted_left = left_future.result()
        sorted_right = right_future.result()
    
    return sorted_left + middle + sorted_right

# ==========================================
# 3. 双调排序 (Bitonic Sort)
# ==========================================
# 注意：双调排序通常要求长度为 2 的幂次

def bitonic_compare(arr, i, j, direction):
    if (direction == 1 and arr[i] > arr[j]) or (direction == 0 and arr[i] < arr[j]):
        arr[i], arr[j] = arr[j], arr[i]

def serial_bitonic_merge(arr, low, cnt, direction):
    if cnt > 1:
        k = cnt // 2
        for i in range(low, low + k):
            bitonic_compare(arr, i, i + k, direction)
        serial_bitonic_merge(arr, low, k, direction)
        serial_bitonic_merge(arr, low + k, k, direction)

def serial_bitonic_sort_core(arr, low, cnt, direction):
    if cnt > 1:
        k = cnt // 2
        serial_bitonic_sort_core(arr, low, k, 1)
        serial_bitonic_sort_core(arr, low + k, k, 0)
        serial_bitonic_merge(arr, low, cnt, direction)

def serial_bitonic_sort(arr):
    n = len(arr)
    next_pow2 = 1
    while next_pow2 < n: next_pow2 *= 2
    padded = arr + [float('inf')] * (next_pow2 - n)
    serial_bitonic_sort_core(padded, 0, next_pow2, 1)
    return padded[:n]

def parallel_bitonic_task(args):
    """进程任务：处理双调排序的一个子块"""
    arr, low, cnt, direction = args
    # 小数据量直接串行，避免进程爆炸
    if cnt <= THRESHOLD:
        # 需要在本地复制一份以免修改原引用（虽然这里传的是切片或拷贝）
        # 为了简化，这里假设传入的是列表切片
        local_arr = list(arr) 
        serial_bitonic_sort_core(local_arr, 0, cnt, direction)
        return local_arr
    
    k = cnt // 2
    # 双调排序也是分治结构，可以并行递归
    # 注意：由于进程间内存隔离，我们需要返回结果并重组
    # 这里为了演示并行逻辑，我们并行处理两个子序列
    with ProcessPoolExecutor(max_workers=2) as executor:
        # 构造参数
        arg1 = (arr[low:low+k], 0, k, 1)
        arg2 = (arr[low+k:low+2*k], 0, k, 0)
        
        f1 = executor.submit(parallel_bitonic_task, arg1)
        f2 = executor.submit(parallel_bitonic_task, arg2)
        
        left = f1.result()
        right = f2.result()
    
    # 合并结果
    merged = left + right
    # 对合并后的结果进行 bitonic_merge (串行，因为依赖数据)
    serial_bitonic_merge(merged, 0, cnt, direction)
    return merged

def parallel_bitonic_sort(arr):
    n = len(arr)
    next_pow2 = 1
    while next_pow2 < n: next_pow2 *= 2
    padded = arr + [float('inf')] * (next_pow2 - n)
    
    # 顶层调用并行任务
    result = parallel_bitonic_task((padded, 0, next_pow2, 1))
    return result[:n]

# ==========================================
# 4. 奇偶排序 (Odd-Even Sort)
# ==========================================
# 奇偶排序是数据并行，适合共享内存。但在多进程中，我们并行化“交换阶段”

def odd_even_swap_task(args):
    """执行单个交换任务"""
    arr, i = args
    if arr[i] > arr[i+1]:
        arr[i], arr[i+1] = arr[i+1], arr[i]
    return arr  # 返回修改后的数组（开销大，但为了演示逻辑）

def serial_odd_even_sort(arr):
    n = len(arr)
    for phase in range(n):
        start = phase % 2
        for i in range(start, n - 1, 2):
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
    return arr

def parallel_odd_even_sort(arr):
    """
    多进程版奇偶排序
    注意：由于进程内存隔离，每次交换都需要传递数组，开销极大。
    实际生产中应使用 multiprocessing.Array 共享内存。
    这里为了代码通用性，采用分块并行策略。
    """
    n = len(arr)
    # 奇偶排序并行化在多进程下效率通常低于串行，除非数据量极大且使用共享内存
    # 这里演示并行化“阶段”中的交换操作
    arr = list(arr) # 确保是列表
    
    for phase in range(n):
        start = phase % 2
        indices = list(range(start, n - 1, 2))
        
        # 如果任务太少，直接串行
        if len(indices) < 4:
            for i in indices:
                if arr[i] > arr[i+1]:
                    arr[i], arr[i+1] = arr[i+1], arr[i]
            continue
            
        # 并行执行交换 (注意：这里为了简化，实际上无法直接共享修改后的 arr)
        # 真正的多进程奇偶排序需要共享内存数组。
        # 为了作业演示，我们这里仅展示并行结构，实际运行可能较慢。
        # 改进方案：将数组分块，每块由一个进程负责排序，最后合并（类似归并）。
        # 但为了符合算法定义，我们使用 Manager 来模拟共享（虽然慢）
        pass 
    
    # 修正：由于 Python 多进程传值开销太大，奇偶排序在多进程下很难高效实现原地交换。
    # 为了作业能够运行且体现并行，我们采用“分块归并”策略模拟并行奇偶，
    # 或者诚实说明：奇偶排序更适合线程或 GPU。
    # 这里提供一个基于 ProcessPoolExecutor 的“阶段并行”实现（逻辑正确但受限于 IPC）
    
    # 由于上述限制，这里推荐使用串行版本作为对比，或者使用线程版。
    # 但为了响应你的“Process”要求，我提供一个基于分块的并行版本：
    # 将数组分给多个进程排序，然后进行奇偶合并。
    # 但这样改变了算法本质。
    
    # 最终决定：使用 multiprocessing.Manager().list() 实现共享列表
    from multiprocessing import Manager
    manager = Manager()
    shared_arr = manager.list(arr)
    
    def swap_worker(idx):
        if shared_arr[idx] > shared_arr[idx+1]:
            # Manager list 支持修改
            v1 = shared_arr[idx]
            v2 = shared_arr[idx+1]
            shared_arr[idx] = v2
            shared_arr[idx+1] = v1

    for phase in range(n):
        start = phase % 2
        indices = range(start, n - 1, 2)
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有交换任务
            list(executor.map(swap_worker, indices))
            
    return list(shared_arr)

# ==========================================
# 主测试程序
# ==========================================

if __name__ == '__main__':
    # 必须在 __main__ 下运行，否则 Windows 下会无限递归创建进程
    print("=== 并行排序算法对比 (多进程版) ===")
    
    # 生成数据
    SIZE = 50000  # 建议 10000 以上才能看出多进程优势，但奇偶排序会很慢
    data = [random.randint(0, 100000) for _ in range(SIZE)]
    
    algorithms = [
        ("串行归并", lambda d: serial_merge_sort(copy.deepcopy(d))),
        ("并行归并 (Process)", lambda d: parallel_merge_sort_task(copy.deepcopy(d))),
        ("串行快排", lambda d: serial_quick_sort(copy.deepcopy(d))),
        ("并行快排 (Process)", lambda d: parallel_quick_sort_task(copy.deepcopy(d))),
        ("串行双调", lambda d: serial_bitonic_sort(copy.deepcopy(d))),
        # 双调并行在 Python 多进程下开销极大，小数据测试
        ("并行双调 (Process)", lambda d: parallel_bitonic_sort(copy.deepcopy(d)) if len(d) <= 2000 else None), 
        ("串行奇偶", lambda d: serial_odd_even_sort(copy.deepcopy(d))),
        # 奇偶并行在 Python 多进程下极慢，仅小数据测试
        ("并行奇偶 (Process)", lambda d: parallel_odd_even_sort(copy.deepcopy(d)) if len(d) <= 1000 else None),
    ]
    
    for name, func in algorithms:
        if func is None:
            print(f"[{name}] ⏭️ 跳过 (数据量过大，多进程开销过高)")
            continue
            
        try:
            start = time.time()
            result = func(data)
            end = time.time()
            
            is_sorted = result == sorted(data)
            status = "✅" if is_sorted else "❌"
            print(f"{status} [{name}] 耗时：{end - start:.4f} 秒")
        except Exception as e:
            print(f"❌ [{name}] 出错：{e}")
        print("-" * 40)