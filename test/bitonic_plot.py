# import matplotlib.pyplot as plt
# import numpy as np

# def draw_comparator(ax, x, y1, y2, direction='up'):
#     """画出比较器，direction决定箭头方向"""
#     if direction == 'up':
#         ax.arrow(x, min(y1, y2), 0, abs(y2-y1), head_width=0.05, head_length=0.1, 
#                 fc='black', ec='black', length_includes_head=True)
#     else:
#         ax.arrow(x, max(y1, y2), 0, -abs(y2-y1), head_width=0.05, head_length=0.1, 
#                 fc='black', ec='black', length_includes_head=True)

# def add_colored_region(ax, x1, x2, y1, y2, color, alpha=0.2):
#     """添加背景色块"""
#     rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, facecolor=color, alpha=alpha)
#     ax.add_patch(rect)

# def draw_bitonic_sorting_network(n=16):
#     fig, ax = plt.subplots(figsize=(15, 8))
#     ax.set_xlim(-0.5, 10)
#     ax.set_ylim(-0.5, n-0.5)
#     ax.axis("off")

#     # 绘制水平线
#     for i in range(n):
#         ax.plot([-0.5, 10], [i, i], 'k', linewidth=0.3, alpha=0.2)

#     # 第一阶段：基础比较（绿色区域）
#     x = 1.0
#     for i in range(0, n, 2):
#         add_colored_region(ax, x-0.3, x+0.3, i-0.3, i+1.3, 'lightgreen', alpha=0.15)
#         # 根据位置决定方向
#         direction = 'up' if (i//2) % 2 == 0 else 'down'
#         draw_comparator(ax, x, i, i+1, direction)

#     # 第二阶段：2-2比较
#     x = 3.0
#     for i in range(0, n, 4):
#         add_colored_region(ax, x-0.3, x+0.3, i-0.3, i+3.3, 'pink', alpha=0.15)
#         add_colored_region(ax, x-0.5, x+0.5, i-0.5, i+3.5, 'lightblue', alpha=0.1)
#         # 第一步，跨距为2的比较
#         direction = 'up' if (i//4) % 2 == 0 else 'down'
#         draw_comparator(ax, x, i, i+2, direction)
#         draw_comparator(ax, x, i+1, i+3, direction)
#         # 第二步，相邻比较
#         x2 = x + 0.8
#         draw_comparator(ax, x2, i, i+1, direction)
#         draw_comparator(ax, x2, i+2, i+3, direction)

#     # 第三阶段：4-4比较
#     x = 5.5
#     for i in range(0, n, 8):
#         add_colored_region(ax, x-0.3, x+0.3, i-0.3, i+7.3, 'pink', alpha=0.15)
#         add_colored_region(ax, x-0.5, x+0.5, i-0.5, i+7.5, 'lightblue', alpha=0.1)
#         # 第一步，跨距为4的比较
#         direction = 'up' if (i//8) % 2 == 0 else 'down'
#         for j in range(i, i+4):
#             draw_comparator(ax, x, j, j+4, direction)
#         # 第二步，跨距为2的比较
#         x2 = x + 0.8
#         for j in range(i, i+8, 4):
#             draw_comparator(ax, x2, j, j+2, direction)
#             draw_comparator(ax, x2, j+1, j+3, direction)
#         # 第三步，相邻比较
#         x3 = x + 1.6
#         for j in range(i, i+8, 2):
#             draw_comparator(ax, x3, j, j+1, direction)

#     # 第四阶段：8-8比较
#     x = 8.0
#     add_colored_region(ax, x-0.3, x+0.3, -0.3, n-0.3, 'pink', alpha=0.15)
#     add_colored_region(ax, x-0.5, x+0.5, -0.5, n-0.5, 'lightblue', alpha=0.1)
#     # 第一步，跨距为8的比较
#     for i in range(0, n//2):
#         draw_comparator(ax, x, i, i+8, 'down')  # 最后阶段所有比较都是降序
#     # 第二步，跨距为4的比较
#     x2 = x + 0.8
#     for i in range(0, n, 4):
#         draw_comparator(ax, x2, i, i+2, 'down')
#         draw_comparator(ax, x2, i+1, i+3, 'down')
#     # 第三步，相邻比较
#     x3 = x + 1.6
#     for i in range(0, n, 2):
#         draw_comparator(ax, x3, i, i+1, 'down')

#     plt.title("Bitonic Sorting Network with 16 Inputs")
#     plt.tight_layout()
#     plt.show()

# # 运行代码
# draw_bitonic_sorting_network(16)

def bitonic_sort(a):
    """
    实现双调排序（升序）
    a: 输入数组（从索引1开始使用，索引0不使用）
    N: 数组长度（必须是2的幂）
    """
    N = len(a) - 1  # 因为我们从索引1开始使用
    
    def print_comparison(k1, k2):
        print(f"线程{k1} 和 线程{k2} 比较: {a[k1]} 和 {a[k2]}")
    
    # 外层循环：控制比较的步长
    i, i2 = 1, 2
    while i2 <= N:
        print(f"\n============= 外层循环 i={i}, i2={i2} =============")
        
        # 第一个内层循环
        j, j2 = 1, i2-1
        while j2 >= 1:
            print(f"\n第一阶段比较 (j={j}, j2={j2}):")
            # 执行比较和交换
            k = j
            while k <= N:
                print_comparison(k, k+j2)
                if a[k] > a[k+j2]:  # 改为大于号，实现升序
                    a[k], a[k+j2] = a[k+j2], a[k]
                k += i2
            j += 1
            j2 -= 2
        
        # 第二个内层循环
        j = i2 >> 2  # j = i2/4
        while j >= 1:
            print(f"\n第二阶段比较 (j={j}):")
            # 对每个子序列进行比较
            for k in range(1, j+1):
                l = k
                while l <= N:
                    print_comparison(l, l+j)
                    if a[l] > a[l+j]:  # 改为大于号，实现升序
                        a[l], a[l+j] = a[l+j], a[l]
                    l += (j << 1)  # l += 2*j
            j >>= 1  # j = j/2
        
        i += 1
        i2 <<= 1  # i2 *= 2

# 测试代码
if __name__ == "__main__":
    # 创建一个长度为17的数组（索引0不使用）
    # arr = [0, 14, 7, 3, 12, 9, 11, 6, 2, 15, 5, 8, 1, 10, 4, 13, 16]
    import random
    arr = [0] + [random.randint(1, 1000) for _ in range(512)]  # 1-1000之间的随机整数
    n = len(arr) - 1
    
    print(f"初始数据: {arr[1:]}")  # 不显示索引0的元素
    
    # 确保数组长度是2的幂
    if n & (n-1) != 0:
        print("数组长度必须是2的幂")
    else:
        bitonic_sort(arr)
        print("\n=====================================")
        print(f"排序结果: {arr[1:]}")  # 不显示索引0的元素
        
        # 验证排序结果（改为检查升序）
        is_sorted = all(arr[i] <= arr[i+1] for i in range(1, len(arr)-1))
        print(f"排序是否正确: {is_sorted}")