import math
import pickle as pkl
import random
import torch
import re
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import time
from sklearn import metrics
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, DBSCAN, KMeans
from scipy.cluster.hierarchy import dendrogram
import seaborn as sb
import pandas as pd

def EmbidToGridid(embd_id):
    if isinstance(embd_id, str):
        embd_id = int(embd_id)
    y = embd_id % 100 #竖轴坐标
    x = ((embd_id - y) / 100) % 100 #横轴坐标
    z = int(embd_id / 10000.0)
    base = (z - 1) * 480
    id = x * 30 +  y + base

    return int(id)

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
            step = [cur_x, cur_y, parseTimestamp(
                tstmp), interval, embeddding_id]
            cur_traj.append(step)

        with open("Day" + str(i) + "_Traj.pkl", "wb") as f:
            pkl.dump(trajs, f)

        with open("Day" + str(i) + "_special_region_entrance.pkl", "wb") as f:
            pkl.dump(stay_idxs, f)

def formatTrajGAILData():
    # 活动节点数：2751 (最大编号为2750)
    # 起始、结束标志：2751, 2752
    from_to_action_pair = {} # 'from to': action
    max_action_list = [0 for i in range(2751)]
    network_str_set = set()
    traj_str_list = []
    traj_count = 0
    step_count = 0

    for day in range(1, 4):
        with open(f"./data/Day{day}_stay.pkl", "rb") as f:
            trajs = pkl.load(f)
        for id in trajs:
            traj_count += 1
            traj = trajs[id]
            traj_len = len(traj)
            for i in range(traj_len):
                step_count += 1
                step = traj[i]
                st_idx = int(step[5])
                traj_str_list.append("{},{},{},{}\n".format(step_count, traj_count, i, st_idx))
                if i < traj_len - 1:
                    next_st_idx = int(traj[i + 1][5])
                    from_to_key = "{} to {}".format(st_idx, next_st_idx)
                    if not from_to_key in from_to_action_pair:
                        action = max_action_list[st_idx]
                        from_to_action_pair[from_to_key] = action
                        network_str_set.add("{} {} {}\n".format(st_idx, action, next_st_idx))
                        max_action_list[st_idx] += 1
    
    network = list(network_str_set)
    network.sort()

    with open("./baselines/TrajGAIL/data/Network.txt", "w") as f:
        f.writelines(network)
    with open("./baselines/TrajGAIL/data/Expert_demo.csv", "w") as f:
        f.write(",oid,eid,stayId\n")
        f.writelines(traj_str_list)
    
    print(max(max_action_list)) # 1333
    # action 13-17: start / end: added when inporting expert demonstrations;

def getVisitPoints(gates):
    # 遍历三天的轨迹(原数据)，记录所有时空节点（空间）。
    # 遍历时，保存当前最大节点idx，保存一个 时空节点代号-idx 的dict （一对一）; 构造 空间节点-idx 字典 （一对多, 不重复），记录每个空间节点包含的时空idx
    # 时空代号：“{空间节点idx}-{时间节点idx}"; 空间节点idx直接从step里面读取，时间节点idx计算得到。
    # 如果没有当前节点，则添加当前节点代号为key，将最大idx加一作为key值，并赋予给当前节点为idx
    # 返回最大idx数，知道多少个节点。

    # step = [cur_x, cur_y, parseTimestamp(tstmp), interval, embeddding_id]
    # 添加时空节点序号的step: step = [cur_x, cur_y, parseTimestamp(tstmp), interval, embeddding_id, st_idx]
    max_idx = 0
    key_idx_pair = {} # {"spatial-temporal": idx}
    s_idx_pair = {} # {spatial-idx(即 embedding-id): set(idx)}
    feature_step_list = [] #[{grid-id:, dur:,}...] feature-node-id -> list-id -> embd-id -> grid-id 
    idx_s_t_tensor = torch.zeros(2751, 2).to(torch.long)
    grid_id_set = set()

    with open("./data/Sensor Distribution Data.csv", "r") as f:
        sensor_info = {}
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(",")
            sensor_info[line[0]] = line[1]

    def _st(step, max_idx):
        emb_id = int(step[4])
        dur = step[3]
        st_key = getSTKey(dur, emb_id)
        st_idx = None
        if st_key in key_idx_pair:
            st_idx = key_idx_pair[st_key]
        else:
            st_idx = max_idx
            key_idx_pair[st_key] = st_idx
            idx_s_t_tensor[st_idx, 0] = emb_id
            idx_s_t_tensor[st_idx, 1] = get_interval_category_(dur)
            grid_id = EmbidToGridid(sensor_info[step[4]])
            feature_step_list.append({'grid-id':grid_id, 'dur':dur})
            grid_id_set.add(grid_id)
            max_idx += 1

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

                    tmp_stays.sort(key=lambda x : x["step_num"])

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
    with open("./data/feature_step_list.pkl", "wb") as f:
        pkl.dump(feature_step_list, f)
    # 2751
    print(max_idx)
    print(feature_step_list)

def writeStayAsCSV():
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

    # todo: feature nodes translate to steps. a dict: {feature node: {id:N, dur:F}}
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
                validate_prob = 1.0
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
    
    with open(f'./data/true.data', 'w') as f:

        for seq_len in day_1_2['train']:
            f.write('{}\n'.format(str(seq_len)))
            seqs = day_1_2['train'][seq_len].numpy().tolist()
            for seq in seqs:
                string = ' '.join([str(s) for s in seq])
                f.write('{}\n'.format(string))

        for seq_len in day_3['train']:
            f.write('{}\n'.format(str(seq_len)))
            seqs = day_3['train'][seq_len].numpy().tolist()
            for seq in seqs:
                string = ' '.join([str(s) for s in seq])
                f.write('{}\n'.format(string))

    with open(f"./data/day_1_2_dataset.pkl", "wb") as f:
        pkl.dump(day_1_2, f)
    with open(f"./data/day_3_dataset.pkl", "wb") as f:
        pkl.dump(day_3, f)

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

def isInPoi(x, y, list_of_places):
    for i in range(len(list_of_places)):
        coords = list_of_places[i]
        if (x - coords[0]) * (x - coords[2]) < 0 and (y - coords[1]) * (y - coords[3]) < 0:
            return i
    return None

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

    tmp = stay_distribution.numpy()
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

    with open("one_step_accessiblity_matrix_.pkl", "wb") as f:
        pkl.dump(one_step_accessiblity_matrix_, f)

def defineMarkovTransitState():
    transit_state = np.zeros((2753, 2753)) # origin-destination
    
    for i in range(1, 3):
        with open("Day" + str(i) + "_stay.pkl", "rb") as f:
            trajs = pkl.load(f)
        for id in trajs:
            traj = trajs[id]
            traj_len = len(traj)
            last_step = 2751
            for idx, step in enumerate(traj):
                step = step[5]
                transit_state[int(last_step), int(step)] += 1.0
                last_step = step
                if idx == traj_len - 1:
                    transit_state[int(step), 2752] += 1.0
    transit_state = transit_state / (np.sum(transit_state, axis=1) + 0.0000001).reshape((2753, 1))
    print(np.sum(transit_state, axis=1))

    with open("./data/Day1_2_transit_state.pkl", 'wb') as f:
        pkl.dump(transit_state, f)

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
        poi_distance = math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))
        return poi_distance / poi_max_distance
    # 两点在不同POI
    return 1

def computeAndStoreSubcost_(grid_list):
    grid_count = len(grid_list)

    subcosts = np.zeros((grid_count, grid_count))
    
    for i in range(grid_count):
        for j in range(grid_count):
            subcosts[i][j] = subcost(grid_list[i], grid_list[j])
    
    with open("./data/subcosts.pkl", "wb") as f:
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
            subcost = subcosts[traj1[i-1]['grid-id']][traj2[j-1]['grid-id']]
            tmp = min(dp[i-1][j-1] + subcost, dp[i][j-1]+1)
            tmp = min(tmp, dp[i][j-1]+1)
            dp[i][j] = tmp

    sedr = dp[M-1][N-1].item()
    return sedr

def workerProcess(traj_count, n1, n2, traj_list, subcosts):
    SEDR_matrix = np.ones((n2 - n1, traj_count)) * -1
    for i in range(n1, n2):
        if i % 100 == 0:
            print(i)
            print(time.localtime(time.time()))
        for j in range(i, traj_count):
            sedr = semanticEDR(traj_list[i], traj_list[j], subcosts)
            SEDR_matrix[i - n1][j] = sedr

    return SEDR_matrix

def computeAndStoreSubcost():
    with open("./data/sensor_info.pkl", "rb") as f:
        sensor_info = pkl.load(f)
    embId_gridId_map = {}
    grid_list = []

    for gridId in sensor_info:
        if gridId == 'sid':
            continue
        sensor = sensor_info[gridId]
        embId = int(sensor['embedding-id'])
        embId_gridId_map[embId] = {}
        embId_gridId_map[embId]['grid-id'] = int(embId)
        embId_gridId_map[embId]['x'] = float(sensor['x'])
        embId_gridId_map[embId]['y'] = float(sensor['y'])
        grid_list.append([float(sensor['x']), float(sensor['y'])])

    computeAndStoreSubcost_(grid_list)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def clusterTrajs(method, num_cluster = 2, squared=False, X=None):
    if method != 'kmeans':
        with open('./data/SEDR_matrix.pkl', 'rb') as f:
            # list; 对称矩阵
            SEDR_matrix = pkl.load(f)

        SEDR_matrix = np.array(SEDR_matrix)
        for i in range(6343):
            SEDR_matrix[i:6343, i] = SEDR_matrix[i, i:6343]

        SEDR_matrix_ = SEDR_matrix.tolist()

    labels = None
    X = X
    
    n_clusters = 0

    if method == 'affinity':
        # similarity 是距离的负数
        similarity = (-SEDR_matrix).tolist()
        preference = -500
        max_iter = 800
        af = AffinityPropagation(affinity='precomputed', preference=preference, max_iter=max_iter).fit(similarity)
        print(af.get_params())

        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_

        n_clusters = len(cluster_centers_indices)

        labels = list(labels)
        label_count = []
        for i in range(n_clusters):
            label_count.append(labels.count(i))

        label_count.sort(reverse=True)
        print('sample count of each cluster: {}'.format(label_count))
        
        with open('./data/cluster_labels_{}_preference{}.pkl'.format(method, preference), 'wb') as f:
            pkl.dump(labels, f)

    elif method == 'agglo':
        model = AgglomerativeClustering(n_clusters=num_cluster, affinity='precomputed', linkage='average', compute_distances='true').fit(SEDR_matrix_)
        labels = model.labels_
        n_clusters = model.n_clusters_
        """
        plt.title("Hierarchical Clustering Dendrogram")
        # plot the top three levels of the dendrogram
        plot_dendrogram(clusters, truncate_mode="level")
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()
        """
        with open('./data/clusters_object_{}_{}.pkl'.format(method, n_clusters), 'wb') as f:
            pkl.dump(model, f)

    elif method == 'dbscan':
        model = DBSCAN(eps=3, min_samples=5, metric='precomputed', n_jobs=-1)
        clustering = model.fit(SEDR_matrix)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        labels = list(labels)
        n_noise = labels.count(-1)
        label_count = []
        for i in range(n_clusters):
            label_count.append(labels.count(i))
        
        print('sample count of each cluster: {}'.format(label_count))
        print('n_noise: {}'.format(n_noise))

    elif method == 'kmeans':
        model = KMeans(n_clusters=num_cluster)
        clustering = model.fit(X)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        labels = list(labels)
        n_noise = labels.count(-1)
        label_count = []
        for i in range(num_cluster):
            label_count.append(labels.count(i))
        
        print('sample count of each cluster: {}'.format(label_count))
        

    """
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print(
        "Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels_true, labels)
    )"""
    if method != 'kmeans':
        print("Estimated number of clusters: %d" % n_clusters)
        print(
            "Silhouette Coefficient: %f"
            % metrics.silhouette_score(SEDR_matrix_, labels, metric="precomputed")
        )
    else:
        print("Estimated number of clusters: %d" % num_cluster)
        print(
            "Silhouette Coefficient: %f"
            % metrics.silhouette_score(X, labels)
        )

    return clustering
    
def evaluateCluster(SEDR_matrix, n_cluster):
    with open('./data/clusters_object_agglo_{}.pkl'.format(n_cluster), 'rb') as f:
        clusters = pkl.load(f)
    
    labels = clusters.labels_
    labels = np.array(labels)
    avg_dist_list = []
    max_size = -1

    for i in range(n_cluster):
        idx_i = np.argwhere(labels == i).reshape(-1)
        size_cluster_i = len(idx_i)
        sum_dist_i = 6343 * 3000
        for j in range(size_cluster_i):
            cur_traj_idx = idx_i[j]
            sum_dist_i = min((np.sum(SEDR_matrix[cur_traj_idx, idx_i]), sum_dist_i))
        
        max_size = max(max_size, size_cluster_i)
        avg_dist_list.append(sum_dist_i * 1.0 / size_cluster_i)

    avg_dist = np.mean(np.array(avg_dist_list))

    print('Agglo cluster {} mean avg_dist: {}, max size: {}'.format(n_cluster, avg_dist, max_size))

def mergeDistance(X, labels, centriods):
    sum_distance = 0
    for i in range(np.size(X, 0)):
        label = labels[i]
        centriod = centriods[label]
        embed = X[i]
        tmp = embed - centriod
        sum_distance += np.sqrt(np.inner(tmp, tmp))
    print('merge distance: {}'.format(sum_distance))

    return sum_distance

def _visualizeCluster(label_path):
    with open("./data/sensor_info.pkl", "rb") as f:
        sensor_info = pkl.load(f)
    embId_gridId_map = {}

    for gridId in sensor_info:
        if gridId == 'sid':
            continue
        sensor = sensor_info[gridId]
        embId = int(sensor['embedding-id'])
        embId_gridId_map[embId] = {}
        embId_gridId_map[embId]['grid-id'] = int(embId)
        embId_gridId_map[embId]['x'] = float(sensor['x'])
        embId_gridId_map[embId]['y'] = float(sensor['y'])

    with open("./data/idx_s_t_tensor.pkl", "rb") as f:
        idx_s_t_tensor = pkl.load(f)
    
    idx_s_t = idx_s_t_tensor.numpy()

    with open('real.data', 'r') as f:
        lines = f.readlines()

    trajs = []
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        if (len(line) == 1):
            continue
        stay_list = [int(i) for i in line]
        traj = []
        for stay in stay_list:
            traj.append(int(idx_s_t[stay][0]))
        trajs.append(traj)

    with open(label_path + '.pkl', 'rb') as f:
        label = np.array(pkl.load(f))

    category_count = np.max(label) + 1

    for c in range(category_count):
        idxs = np.argwhere(label == c).reshape(-1)
        stay_count_array = np.zeros((32, 30))
        for i in idxs:
            traj = trajs[int(i)]
            for s in traj:
                embId = int(idx_s_t[s, 0])
                grid = embId_gridId_map[embId]
                x = math.floor(float(grid['x']))
                y = math.floor(float(grid['y']))
                stay_count_array[x, y] += 1

        idx = [i for i in range(32)]
        col = [i for i in range(30)]
        data = pd.DataFrame(stay_count_array, index=idx, columns=col)
        plt.figure(dpi=200, figsize=(15,12))
        sb.heatmap(data, cmap='Reds', robust=True)
        plt.savefig("{}_cluster{}_heatmap.png".format(label_path, c))

        #plt.show()

def visualizeCluster():
    label_path = './data/cluster_labels_affinity_preference-300'
    _visualizeCluster(label_path)

if __name__ == '__main__':
    
    # 聚类
    method = 'kmeans'
    m = []

    with open('./data/traj_embedding_numpy_array.pkl', 'rb') as f:
        X = pkl.load(f)
    
    for i in range(2, 10):
        clustering = clusterTrajs(method=method, num_cluster=i, X=X)
        with open('./data/cluster_labels{}.pkl'.format(i), 'wb') as f:
            pkl.dump(clustering.labels_, f)
        m.append(mergeDistance(X, clustering.labels_, clustering.cluster_centers_))
        print('\n')

    """
    x_axis_data = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] #x
    y_axis_data = m #y

    plt.plot(x_axis_data, y_axis_data, 'o', linewidth=1)#'bo-'表示蓝色实线，数据点实心原点标注
    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

    plt.legend()  #显示上面的label
    plt.xlabel('number of clusters') #x_label
    plt.ylabel('merge distance')#y_label
    
    #plt.ylim(-1,1)#仅设置y轴坐标范围
    plt.show()
    """

    """
    for i in range(2, 101):
        num_cluster = i
        clusterTrajs(method, num_cluster)
    
    # 聚类评估
    with open('./data/SEDR_matrix.pkl', 'rb') as f:
        # list; 对称矩阵
        SEDR_matrix = pkl.load(f)

    SEDR_matrix = np.array(SEDR_matrix)
    for i in range(6343):
        SEDR_matrix[i:6343, i] = SEDR_matrix[i, i:6343]

    for i in range(2, 101):
        evaluateCluster(SEDR_matrix, i)
    """
    #visualizeCluster()