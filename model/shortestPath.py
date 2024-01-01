import numpy as npy

current_path = []
shortest_path = []

min_dist = 1000000

def dfs(od, N, cur, des, book, dist):
    """
    map: od矩阵, 960 * 960, 1/0矩阵
    N: 节点数, 1, val:960
    cur: 当前节点
    des: 目标节点
    book: 已经经过的节点, list[960]
    dist: 当前距离
    min_dist: 最短距离
    """
	# cur 是当前所在城市的编号 - 1
	# dist 是当前已经走过的路程
	# 如果当前走过的路程己经大于之前找到的最短路程，则没有必要再往下尝试了，立即返回
    global min_dist
    global shortest_path

    if dist > min_dist:
        return

	# 判断是否到达了目标城市
    if cur == des:
        if (dist < min_dist):
            min_dist = dist
            shortest_path = current_path.copy()
            print(shortest_path)
        return

	# 从 1 (1 - 1) 号城市到 N (N - 1) 号城市的尝试
    for k in range(N):
		# 判断当前城市 cur 到城市 k 是否有路，并判断城市 k 是否在已经走过的路径中
        if ((0 == book[k]) and (od[cur][k] != 0)):
            book[k] = 1 # 标记城市 k 已经在路径中
            current_path.append(k)
            dfs(od, N, k, des, book, dist + od[cur][k]) # 从城市 k 再出发，继续寻找目标城市
            book[k] = 0 # 之前一步探索完毕之后，取消对城市 k 的标记
            current_path.pop()
    return

if __name__ == "__main__":

    # 重读od
    od = npy.zeros((960, 960), dtype=npy.int32)

    with open('data/connection_matrix.csv', 'r') as f:
        road_count = 0
        for line in f.readlines():
            l = line.split(',')
            for i in range(960):
                od[road_count][i] = int(l[i])
            road_count += 1           

    ret = 0
    cur = 0
    des = 0
    book = npy.zeros(960)
    dist = 0
	
    # 修改测试编号
    cur = 30 * 2 - 1 + 2
    # des = 30 * 16 - 1 + 27
    # des = 30 * 3 - 1 + 8
    des = 70
    book[cur] = 1
    current_path.append(cur)

    dfs(od, 960, cur, des, book, dist)

    ret = min_dist
    print(ret)
    print(shortest_path)