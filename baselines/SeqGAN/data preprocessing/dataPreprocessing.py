import math
import pickle as pkl
import random
import torch
import re
import numpy as np

# 要干嘛呢？

# 1. 时间间隔函数重定义
# 0-30 为第0级
# 30-180 为第1级
# 然后每10分钟（600）一级，上不封顶

# 2. 节点数据重定义
# 遍历三天的轨迹(原数据)，记录所有时空节点（空间）。
# 遍历时，保存当前最大节点idx，保存一个节点时空代号-idx的dict; 构造空间节点-idx字典，记录每个空间节点包含的时空idx
# 时空代号：“{空间节点idx}-{时间节点idx}"; 空间节点idx直接从step里面读取，时间节点idx计算得到。
# 如果没有当前节点，则添加当前节点代号为key，将最大idx加一作为key值，并赋予给当前节点为idx
# 返回最大idx数，知道多少个节点。尽量少一些呜呜呜。

# 3. 可达性矩阵重定义
# 检查原可达性矩阵
# 构造 M * M 新可达性矩阵
# 遍历原可达性矩阵，对于每一个元素，查询其下标在空间节点-时空idx字典中的值，双for循环，用1填充新可达性矩阵

# 4. 构造新数据集
# 去头尾?

# 5. 实时热度
# M * 860 需要归一化，即重整到0-1范围

def get_interval_category_(interval):
    if interval < 30:
        return 0
    if interval < 180:
        return 1
    return 2 + math.floor((interval - 180)*1.0/600)

def get_interval_category(duration_str):
    dur = int(duration_str)
    if dur < 10740:
        return round(dur * 1.0 / 60)
    else:
        return 179

def parseTimestamp(timestamp):
    timestamp = int(timestamp)
    hours = str(int(timestamp / 3600))
    minute = str(int(timestamp / 60 % 60))
    second = str(math.ceil(timestamp % 60))
    if int(minute) < 10 : minute = "0" + minute
    if int(second) < 10 : second = "0" + second
    return hours + ":" + minute + ":" + second

#   l1 [xa, ya, xb, yb]   l2 [xa, ya, xb, yb]
def intersect(l1, l2):
    if (None in l1): return False
    if (None in l2): return False

    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    a = v0[0] * v1[1] - v0[1] * v1[0]
    b = v0[0] * v2[1] - v0[1] * v2[0]

    temp = l1
    l1 = l2
    l2 = temp
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    c = v0[0] * v1[1] - v0[1] * v1[0]
    d = v0[0] * v2[1] - v0[1] * v2[0]

    if a*b < 0 and c*d < 0:
        return True
    else:
        return False

# 先遍历，记录进出region的index，再次遍历，添加驻点和region驻点
# 需要先统计各个区域
toilet3 = [ 20, 10, 22, 10]
toilet2 = [14, 27, 14, 29]
toilet1 = [4, 10, 6, 10]
exhibition = [12, 15, 12, 19]
dining_room = [18, 6, 26, 6]
relaxation_area = [29, 6, 31, 6]

special_poi_entrance = [toilet1, toilet2, toilet3, exhibition, dining_room, relaxation_area]

lift2_1 = {"1": 1, "2": 1}
lift2_2 = {"338": 1, "339": 1}

lift1_1 = {"303": 1, "304": 1}
lift1_2 = {"462": 1, "463": 1}

def formatTrajs():
    with open("./data/Sensor Distribution Data.csv", "r") as f:
        sensor_info = {}
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(",")
            sensor_info[line[1]] = {
                "embedding-id": line[0],
                "floor": line[2],
                "x": line[3],
                "y": line[4]
            }

    with open("./data/sensor_info.pkl", "wb") as f:
        pkl.dump(sensor_info, f)
    
    for i in range(1, 4):
        
        file_path = "./data/Day" + str(i) + ".csv"
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        trajs = {}
        stay_idxs = {}
        pre_x = None
        pre_y = None
        cur_ID = None
        step_count = 0
        
        for j in range(1, len(lines)):
            step_count += 1
            cur_traj = None
            cur_stay = None
            line = lines[j].strip().split(",")
            if (cur_ID and line[0] == cur_ID):
                cur_traj = trajs[cur_ID]
                cur_stay = stay_idxs[cur_ID]
            else:
                cur_ID = line[0]
                cur_traj = trajs[cur_ID] = []
                cur_stay = stay_idxs[cur_ID] = []
            
            tstmp = line[2]
            interval = None

            cur_x = float(sensor_info[line[1]]["x"])
            cur_y = float(sensor_info[line[1]]["y"])
            
            for r in range(len(special_poi_entrance)):
                if intersect([pre_x, pre_y, cur_x, cur_y], special_poi_entrance[r]):
                    cur_stay.append((r, step_count))
            

            interval = 7
            pre_y = pre_x = None

            if j < len(lines) - 1:
                next_line = lines[j + 1].strip().split(",")
                next_id = next_line[0]
                next_tstmp = next_line[2]
                if cur_ID == next_id:
                    interval = int(next_tstmp) - int(tstmp)
                    pre_x = cur_x
                    pre_y = cur_y
                else:
                    step_count = 0
            
            embeddding_id = sensor_info[line[1]]["embedding-id"]
            step = [cur_x, cur_y, parseTimestamp(tstmp), interval, embeddding_id]
            cur_traj.append(step)

        with open("Day" + str(i) + "_Traj.pkl", "wb") as f:
            pkl.dump(trajs, f)

        with open("Day" + str(i) + "_special_region_entrance.pkl", "wb") as f:
            pkl.dump(stay_idxs, f)

def swap(array, i, j):
    tmp = array[i]
    array[i] = array[j]
    array[j] = tmp

def _partition(array, left, right):
    pivot = left
    index = pivot + 1
    for i in range(index, right + 1):
        if array[i]["step_num"] < array[pivot]["step_num"]:
            swap(array, i, index)
            index += 1
    swap(array, pivot, index - 1)
    return index - 1

def _quickSort(array, left, right):
    if left < right:
        pivot = _partition(array, left, right)
        _quickSort(array, left, pivot - 1)
        _quickSort(array, pivot + 1, right -1)

def quickSort(array):
    if len(array) < 2: return
    _quickSort(array, 0, len(array)-1)
    return

def getDistance(step, line):
    x = step[0]
    y = step[1]
    coord = None
    base = None
    if line[1] == line[3]:
        coord = y
        base = line[1]
    else:
        coord = x
        base = line[0]
    return abs(coord - base)

def getSTKey(interval, embedding_id):
    return str(embedding_id) + '-' + str(get_interval_category_(interval))

def getVisitPoints(gates):
    # 遍历三天的轨迹(原数据)，记录所有时空节点（空间）。
    # 遍历时，保存当前最大节点idx，保存一个 时空节点代号-idx 的dict （一对一）; 构造 空间节点-idx 字典 （一对多, 不重复），记录每个空间节点包含的时空idx
    # 时空代号：“{空间节点idx}-{时间节点idx}"; 空间节点idx直接从step里面读取，时间节点idx计算得到。
    # 如果没有当前节点，则添加当前节点代号为key，将最大idx加一作为key值，并赋予给当前节点为idx
    # 返回最大idx数，知道多少个节点。尽量少一些呜呜呜。

    # step = [cur_x, cur_y, parseTimestamp(tstmp), interval, embeddding_id]
    # 添加时空节点序号的step: step = [cur_x, cur_y, parseTimestamp(tstmp), interval, embeddding_id, st_idx]
    max_idx = 0
    key_idx_pair = {} # {"spatial-temporal": idx}
    s_idx_pair = {} # {spatial-idx(即 embedding-id): set(idx)}
    idx_s_t_tensor = torch.zeros(2751, 2).to(torch.long)

    def _st(step, max_idx):
        emb_id = int(step[4])
        dur = step[3]
        st_key = getSTKey(dur, emb_id)
        st_idx = None
        if st_key in key_idx_pair:
            st_idx = key_idx_pair[st_key]
        else:
            st_idx = max_idx
            max_idx += 1
            key_idx_pair[st_key] = st_idx
            idx_s_t_tensor[st_idx, 0] = emb_id
            idx_s_t_tensor[st_idx, 1] = get_interval_category_(dur)
        if emb_id in s_idx_pair:
            s_idx_pair[emb_id].add(st_idx)
        else:
            s_idx_pair[emb_id] = {st_idx}
        step.append(st_idx)
        return max_idx

    for i in range(1, 4):
        with open("./data/Day" + str(i) + "_Traj.pkl", "rb") as f:
            trajs = pkl.load(f)
        with open("./data/Day" + str(i) + "_special_region_entrance.pkl", "rb") as f:
            idxs = pkl.load(f)
        
        stays = {}
        
        for id in trajs:
            traj = trajs[id]
            idx = idxs[id]
            double_entrance_flag = False
            if len(idx) % 2 == 0:
                double_entrance_flag = True

            stay = []
            come_in = None
            come_out = None
            if double_entrance_flag and len(idx) > 0:
                come_in = idx.pop(0)
                come_out = idx.pop(0)
            
            step_num = 0
            pre_embedding_id = None
            while step_num < len(traj):
                
                step = traj[step_num]
                
                # 进出口加入逗留点
                if step_num == 0 or step_num == len(traj) - 1:
                    max_idx = _st(step, max_idx)
                    stay.append(step)
                    step_num += 1
                    pre_embedding_id = step[4]
                    continue

                # 如果entrance无奇数次，且当前步数等于come_in步数
                if double_entrance_flag and come_in and step_num == come_in[1] - 1:
                    tmp_stays = []
                    tmp_dis = 0
                    far_point = None
                    
                    while step_num < come_out[1] - 1:
                        duration = step[3]
                        distance = getDistance(step, gates[come_in[0]])
                        if duration > 180:
                            tmp_stays.append({"step": step, "step_num": step_num})
                            if distance > tmp_dis: 
                                tmp_dis = distance
                                far_point = None
                        elif distance > tmp_dis:
                            tmp_dis = distance
                            far_point = {"step": step, "step_num": step_num}
                        step_num += 1
                        step = traj[step_num]
                        pre_embedding_id = step[4]

                    if far_point != None:
                        tmp_stays.append(far_point)

                    quickSort(tmp_stays)

                    for point in tmp_stays:
                        max_idx = _st(point["step"], max_idx)
                        stay.append(point["step"])
                
                    come_in = None
                    come_out = None
                    if len(idx) > 0:
                        come_in = idx.pop(0)
                        come_out = idx.pop(0)
                
                duration = step[3]
                _flag = False
                if duration > 180:
                    _flag = True
                elif pre_embedding_id in lift1_1 and step[4] in lift1_2:
                    _flag = True
                elif pre_embedding_id in lift1_2 and step[4] in lift1_1:
                    _flag = True
                elif pre_embedding_id in lift2_1 and step[4] in lift2_2:
                    _flag = True
                elif pre_embedding_id in lift2_2 and step[4] in lift2_1:
                    _flag = True

                if _flag:
                    max_idx = _st(step, max_idx)
                    stay.append(step)

                step_num += 1
                pre_embedding_id = step[4]

            stays[id] = stay
        
        with open("Day" + str(i) + "_stay.pkl", "wb") as f:
            pkl.dump(stays, f)
    
    with open("./data/key_idx_pair.pkl", "wb") as f:
        pkl.dump(key_idx_pair, f)
    with open("./data/s_idx_pair.pkl", "wb") as f:
        pkl.dump(s_idx_pair, f)
    with open("./data/idx_s_t_tensor.pkl", "wb") as f:
        pkl.dump(idx_s_t_tensor, f)
    # 2751
    print(max_idx)

# formatTrajs()
# getVisitPoints(special_poi_entrance)
with open("./data/s_idx_pair.pkl", "rb") as f:
    s_idx_pair = pkl.load(f)
ss = [265, 322, 323, 325]
for s in ss:
    print(s_idx_pair[s])
pass

def writeCSV():
    for day in range(1, 4):
        with open("Day" + str(day) + "_stay.pkl", "rb") as f:
            stays = pkl.load(f)
        string = ""
        for id in stays:
            stay = stays[id]
            for step in stay:
                x = str(step[0])
                y = str(step[1])
                time = str(step[2])
                interval = str(step[3])
                string += id + "," + x + "," + y + "," + time + "," + interval + "\n"
        
        with open("Day" + str(day) + "_stay.csv", "w") as f:
            f.write(string)

def buildDataset():
    # day1 day2 合在一起，day3 单独做一个数据集
    # 一个数据集为一个对象，有三个key，其值为train, validate, test集;

    # 训练集以len为key, 值为一个数组, 数组元素由两个key构成, 一个是start_tstmp, 另一个是stay;
    # 训练、验证、测试集分割:
    # day1、day2: 8 : 1 : 1
    # day3: 1 : 1 : 1
    # 按概率分配, 随机数种子为 0 
    
    # 验证集和测试集为数组, 按任意顺序排列

    # build day 1 2 dataset
    day_1_2 = {'train':{}, 'validate':[], 'test':[]}
    day_3 = {'train':{}, 'validate':[], 'test':[]}

    random.seed(0)
    for day in range(1, 4):
        with open(f"./data/Day{day}_stay.pkl", "rb") as f:
            trajs = pkl.load(f)
        for id in trajs:
            traj = trajs[id]
            traj_len = len(traj)
            tmp_traj = []
            for i in range(traj_len):
                step = traj[i]
                st_idx = int(step[5])
                tmp_traj.append(st_idx)
            tmp_traj = torch.as_tensor(tmp_traj)
            train_prob = 0
            validate_prob = 0
            if day == 1 or day == 2:
                dataset = day_1_2
                train_prob = 0.8
                validate_prob = 0.9
            else:
                dataset = day_3
                train_prob = 0.33333
                validate_prob = 0.66667
            rand = random.random()
            key = ""
            if rand < train_prob:
                key = "train"
            elif rand < validate_prob:
                key = "validate"
            else:
                key = "test"
            if key == "train":
                if traj_len in dataset[key]:
                    dataset[key][traj_len].append(tmp_traj)
                else:
                    dataset[key][traj_len] = [tmp_traj]
            else:
                dataset[key].append(tmp_traj)
    
    for traj_len in day_1_2["train"]:
        day_1_2["train"][traj_len] = torch.stack(day_1_2["train"][traj_len], dim=0)
    for traj_len in day_3["train"]:
        day_3["train"][traj_len] = torch.stack(day_3["train"][traj_len], dim=0)

    with open(f"./data/day_1_2_dataset.pkl", "wb") as f:
        pkl.dump(day_1_2, f)
    with open(f"./data/day_3_dataset.pkl", "wb") as f:
        pkl.dump(day_3, f)
        
# buildDataset()
# getStayPoints(special_regions)
# writeCSV()

def formatStayPoints():

    for day in range(1, 4):
        
        stay = {}
        start_time = {}

        with open("./data/Day" + str(day) + "_stay.pkl", "rb") as f:
            stays = pkl.load(f)
        
        stay_lens = {}

        for id in stays:
            traj = stays[id]
            stay_len = len(traj)
            stay_len_str = str(stay_len)
            stay_tensor = torch.zeros(stay_len, 2)
            for i in range(stay_len):
                stay_tensor[i][0] = int(traj[i][4])
                stay_tensor[i][1] = get_interval_category(traj[i][3])
            stay_tensor.to(torch.int64)
            if stay_len_str in stay_lens:
                stay_lens[stay_len_str] += 1
                stay[stay_len_str].append(stay_tensor)
                start_time[stay_len_str].append(traj[0][2])
            else:
                stay_lens[stay_len_str] = 1
                stay[stay_len_str] = [stay_tensor]
                start_time[stay_len_str] = [traj[0][2]]
        
        data = {
            "stay": stay,
            "start_time": start_time,
            "stay_lens": stay_lens
        }

        with open("Day" + str(day) + "_data.pkl", "wb") as f:
            pkl.dump(data, f)

# formatStayPoints()

# 以15分钟为间隔计算人员密度
"""
sensor_info[line[1]] = {
"embedding-id": line[0],
"floor": line[2],
"x": line[3],
"y": line[4]
}
"""

sub_venue_a = [2, 1, 4, 6]
sub_venue_b = [4, 1, 6, 6]
sub_venue_c = [6, 1, 8, 6]
sub_venue_d = [8, 1, 10, 6]
register = [12, 2, 14, 6]
poster_area = [3, 7, 10, 9]
toilet_1 = [4, 10, 6, 12]
tea_break_1 = [6, 10, 10, 12]
vip_lounge = [10, 10, 12, 12]
lift_1_1 = [14, 10, 15, 12]
lift_2_1 = [1, 10, 2, 12]
exhibition_hall = [2, 15, 12, 19]
main_venue = [2, 19, 12, 29]
service = [14, 19, 16, 21]
tea_break_2 = [14, 21, 16, 25]
media_room = [14, 25, 16, 27]
toilet_2 = [14, 27, 16, 29]
lift_2_2 = [17, 10, 19, 12]
lift_1_2 = [30, 10, 31, 12]
toilet_3 = [20, 10, 22, 12]
work_room = [22, 10, 24, 12]
dining_room = [18, 1, 26, 6]
hacking_contest_room = [26, 1, 28, 6]
relaxation_area = [29, 0, 32, 6]

list_of_pois = [sub_venue_a, sub_venue_b, sub_venue_c, sub_venue_d, register, poster_area, toilet_1, tea_break_1, vip_lounge, lift_1_1, lift_2_1, exhibition_hall, main_venue, service, tea_break_2, media_room, toilet_2, lift_2_2, lift_1_2, toilet_3, work_room, dining_room, hacking_contest_room, relaxation_area]

entrance_0 = [13, 0, 14, 1] # 265
entrance_1 = [15, 2, 16, 3] # 322
entrance_2 = [15, 4, 16, 5] # 323
entrance_3 = [15, 7, 16, 8] # 325

list_of_entrances = [entrance_0, entrance_1, entrance_2, entrance_3]

exit_0 = [15, 5, 16, 6]
exit_1 = [15, 15, 16, 16]
exit_2 = [15, 17, 16, 18]
exit_3 = [0, 19, 1, 20]

list_of_exits = [exit_0, exit_1, exit_2, exit_3]


def isInPoi(x, y, list_of_places):
    for i in range(len(list_of_places)):
        coords = list_of_places[i]
        if (x - coords[0]) * (x - coords[2]) < 0 and (y - coords[1]) * (y - coords[3]) < 0:
            return i
    return None

list_of_all_places = list_of_pois + list_of_entrances + list_of_exits

# 计算subcost
def subcost(p1, p2) :
    # 两点位置相同
    if (p1[0] == p2[0]) and (p1[1] == p2[1]):
        return 0
    poi1 = isInPoi(p1[0], p1[1], list_of_all_places)
    poi2 = isInPoi(p2[0], p2[1], list_of_all_places)
    # 两点在同一POI且位置不同
    if (poi1 and poi2 and poi1 == poi2):
        poi_extent = list_of_all_places[poi1]
        poi_max_distance = math.sqrt(math.pow(poi_extent[0] - poi_extent[2], 2) + math.pow(poi_extent[1] - poi_extent[3], 2))
        poi_distance = math.sqrt(math.pow(poi1[0] - poi2[0], 2) + math.pow(poi1[1] - poi2[1], 2))
        return poi_distance / poi_max_distance
    # 两点在不同POI
    return 1

def computeAndStoreSubcost(grid_list):
    grid_count = len(grid_list)

    subcosts = np.zeros((grid_count, grid_count))
    
    for i in range(grid_count):
        for j in range(grid_count):
            subcosts[i][j] = subcost(grid_list[i], grid_list[j])
    
    with open("subcosts.pkl", "wb") as f:
        pkl.dump(subcosts, f)

def semanticEDR(traj1, traj2, subcosts):
    # 建立dp矩阵, 在subcosts 01矩阵上做动态规划找一条最短路径即可
    M = len(traj1) + 1
    N = len(traj2) + 1
    dp = np.zeros((M, N))

    # dp 初始化
    dp[0][0] = 0
    for n in range(1, N) :
        dp[0][n] = n
    for m in range(1, M) :
        dp[m][0] = m

    # 递推
    for i in range(1, M) :
        for j in range(1, N) :
            min = math.min(dp[i-1][j-1]+subcosts[i][j], dp[i][j-1]+1)
            min = math.min(min, dp[i][j-1]+1)
            dp[i][j] = min
    
    return dp[M-1][N-1]

def computeAndStoreSEDR_impl(traj_list):
    traj_count = len(traj_list)
    SEDR_matrix = np.ones((traj_count, traj_count)) * -1
    with open('subcosts.pkl', "rb") as f:
        subcosts = pkl.load(f)
    for i in range(traj_count):
        for j in range(i, traj_count):
            sedr = semanticEDR(traj_list[i], traj_list[j], subcosts)
            SEDR_matrix[i][j] = sedr
    with open("semanticEDR.pkl", "wb") as f:
        pkl.dump(SEDR_matrix, f)

def computeAndStoreEDR():
    """
    sensor_info[line[1]] = {
    "embedding-id": line[0],
    "floor": line[2],
    "x": line[3],
    "y": line[4]
    }
    """
    """
    idx_s_t_tensor = torch.zeros(2751, 2).to(torch.long)
    idx_s_t_tensor[st_idx, 0] = emb_id
    idx_s_t_tensor[st_idx, 1] = get_interval_category_(dur)
    """
    pass

def clusterTrajs():
    pass

def timeToSeconds(time_str):
    s = 0
    time_triple = time_str.split(":")
    for i in range(3):
        s += int(re.sub("^0", '', time_triple[i])) * math.pow(60, 2 - i)
    return s

def getTimeSlot(day, time_str):
    s = (day - 1) * 24 * 3600 + timeToSeconds(time_str)
    return int(s / 300)

def getVisitorDistribution():
    
    # 864 = 3d * 24h * 3600s / (5m * 60s)
    # 入口分布：用于生成轨迹起始点；(4 * 864)
    # 出口分布：可能用于出口分配，但是不一定需要分配，看实验情况 (4 * 864)
    # 各地stay时长分布（用于时间掩膜：删去过长的停留，如在厕所待了2个小时，在电梯里待5分钟以上等）：24 * 180（0-180分钟的每分钟分布）(tensor)
    # 以5分钟为时间间隔计算各个POI人员密度即在当下时段在场地“活动”的人数（即在场地有stay点的人数）: 24 * 864

    # step = [cur_x, cur_y, parseTimestamp(tstmp), interval, embeddding_id, st_idx]

    entrance_distribution = torch.zeros(4, 864).to(torch.int64)
    exit_distribution = torch.zeros(4, 864).to(torch.int64)
    # stay_dur_distribution = torch.zeros(25, 180).to(torch.float32)
    stay_distribution = torch.zeros(2751, 864).to(torch.float32)
    start_time_and_traj_lens_joint_distribution = torch.zeros(864, 60, dtype=torch.int64)
    lens_distribution = torch.zeros(60, dtype=torch.int64)

    for i in range(1, 4):
        with open("./data/Day" + str(i) + "_stay.pkl", "rb") as f:
            stay_trajs = pkl.load(f)

        for id in stay_trajs:
            traj = stay_trajs[id]

            start_point = traj[0]
            entrance_idx = isInPoi(start_point[0], start_point[1], list_of_entrances)
            time_slot = getTimeSlot(i, start_point[2])
            traj_len = len(traj)
            start_time_and_traj_lens_joint_distribution[time_slot, traj_len] += 1
            lens_distribution[traj_len] += 1
            entrance_distribution[entrance_idx, time_slot] += 1

            end_point = traj[traj_len - 1]
            exit_idx = isInPoi(end_point[0], end_point[1], list_of_exits)
            time_slot = getTimeSlot(i, end_point[2])
            exit_distribution[exit_idx, time_slot] += 1


            for j in range(len(traj)):
                stay = traj[j]
                if j == 0 or j == len(traj) - 1:
                    continue
                stay_distribution[stay[5], getTimeSlot(i, stay[2])] += 1
                
                """
                poi_idx = isInPoi(stay[0], stay[1], list_of_pois)
                if poi_idx == None:
                    if get_interval_category(stay[3]) > 2:
                        stay_place_distribution[24, getTimeSlot(i, stay[2])] += 1
                        #stay_dur_distribution[24, get_interval_category(stay[3])] += 1
                    continue
                stay_place_distribution[poi_idx, getTimeSlot(i, stay[2])] += 1
                #stay_dur_distribution[poi_idx, get_interval_category(stay[3])] += 1
                """

    for i in range(stay_distribution.size(dim=1)):
        max = torch.max(stay_distribution[:, i])
        if max.item() == 0:
            stay_distribution[:, i] = 1.0
        else: 
            stay_distribution[:, i] = torch.true_divide(stay_distribution[:, i], max/2)

    with open("./data/entrance_distribution.pkl", "wb") as f:
        pkl.dump(entrance_distribution, f)
    with open("./data/exit_distribution.pkl", "wb") as f:
        pkl.dump(exit_distribution, f)
    #with open("./data/stay_dur_distribution.pkl", "wb") as f:
        #pkl.dump(stay_dur_distribution, f)
    with open("./data/stay_distribution.pkl", "wb") as f:
        pkl.dump(stay_distribution, f)
    with open("./data/start_time_and_traj_lens_joint_distribution.pkl", "wb") as f:
        pkl.dump(start_time_and_traj_lens_joint_distribution, f)
    with open("./data/lens_distribution.pkl", "wb") as f:
        pkl.dump(lens_distribution, f)


def getAverageInBetweenDuration():
    count = 0
    total_dur = 0
    for i in range(1, 4):
        with open("./data/Day" + str(i) + "_stay.pkl", "rb") as f:
            stay_trajs = pkl.load(f)
        for id in stay_trajs:
            traj = stay_trajs[id]
            pre_time = None
            for stay in traj:
                if pre_time == None:
                    pre_time = timeToSeconds(stay[2]) + float(stay[3])
                else:
                    cur_time = timeToSeconds(stay[2])
                    total_dur += (cur_time - pre_time) / 1000.0
                    count += 0.001
                    pre_time = cur_time + float(stay[3])
    
    return total_dur / count

# print(getAverageInBetweenDuration()) # 176.73519403587665

def defineOneStepAccessibilityMatrix():

    one_step_accessiblity_matrix = torch.zeros(470, 470) # origin-destination

    # a place on one floor (except for lifts) are one-step accessible to all the place on the same floor
    # floor 1
    for i in range(338):
        for j in range(338):
            if j in [1, 2, 303, 304] or i == j:
                continue
            one_step_accessiblity_matrix[i, j] = 1
    # floor 2 
    for i in range(338, 470):
        for j in range(338, 470):
            if j in [338, 339, 462, 463] or i == j:
                continue
            one_step_accessiblity_matrix[i, j] = 1
    
    # lifts on one floor are accessible to places on ajdacent floors
    # lift_1_1 and lift_2_1
    for i in range(338, 470):
        if i in [338, 339, 462, 463]:
            continue
        for j in [1, 2, 303, 304]:
            one_step_accessiblity_matrix[i, j] = 1
    # lift_1_2 and lift_2_2
    for i in range(338):
        if i in [1, 2, 303, 304]:
            continue
        for j in [338, 339, 462, 463]:
            one_step_accessiblity_matrix[i, j] = 1
    
    one_step_accessiblity_matrix_ = torch.zeros(2751, 2751)

    with open("./data/s_idx_pair.pkl", "rb") as f:
        s_idx_pair = pkl.load(f)

    count = 0
    for entry in s_idx_pair:
        count += len(s_idx_pair[entry])

    for i in range(470):
        for j in range(470):
            if not ((i in s_idx_pair) and (j in s_idx_pair)):
                continue
            starts = s_idx_pair[i]
            ends = s_idx_pair[j]
            for start in starts:
                for end in ends:
                    if one_step_accessiblity_matrix[i, j] == 1:
                        one_step_accessiblity_matrix_[start, end] = 1

    tmp = one_step_accessiblity_matrix_.numpy() 
    sum = torch.sum(one_step_accessiblity_matrix_) / 2751.0
    with open("one_step_accessiblity_matrix_.pkl", "wb") as f:
        pkl.dump(one_step_accessiblity_matrix_, f)


def getPoiSID():
    with open("./data/sensor_info.pkl", "rb") as f:
        sensor_info = pkl.load(f)
    poi_eid = []
    poi_eid_tensor = torch.zeros(25, 470)
    eid_poi = torch.ones(470).to(torch.int64) * 24
    for i in range(len(list_of_pois)):
        floor = 1
        poi = list_of_pois[i]
        eids = []
        if poi[0] > 16:
            poi[0] -= 16
            poi[2] -= 16
            floor = 2
        for x in range(poi[0], poi[2]):
            for y in range(poi[1], poi[3]):
                str_x = str(x)
                str_y = str(y)
                if x < 10:
                    str_x = "0" + str_x
                if y < 10:
                    str_y = "0" + str_y
                sid = str(floor) + str_x + str_y
                if sid in sensor_info:
                    eid = sensor_info[sid]["embedding-id"]
                    eids.append(int(eid))
                    poi_eid_tensor[i, int(eid)] = 1
                    eid_poi[int(eid)] = i
        poi_eid.append(eids)
    
    sum = torch.sum(poi_eid_tensor, dim=0)
    walking_area = 1 - sum
    poi_eid_tensor[24, :] = walking_area

    with open("./data/poi_eid.pkl", "wb") as f:
        pkl.dump(poi_eid, f)
    with open("./data/poi_eid_tensor.pkl", "wb") as f:
        pkl.dump(poi_eid_tensor, f)
    with open("./data/walking_area_eid.pkl", "wb") as f:
        pkl.dump(walking_area, f)
    with open("./data/eid_poi.pkl", "wb") as f:
        pkl.dump(eid_poi, f)