import os,sys

sys.path.append("./")
from dataProcessing.utils import *
import pickle as pkl
import math
import torch

def formatTrajsAsImg():
    with open("./data/sensor_info.pkl", "rb") as f:
        sensor_info = pkl.load(f)
    traj_id = 0

    for day in range(1, 4):
        file_path = "./data/Day" + str(day) + ".csv"
        with open(file_path, 'r') as f:
            lines = f.readlines()
        first_line = lines[1].strip().split(",")
        cur_id = first_line[0]
        start_idx = 1
        end_idx = 2

        for i in range(2, len(lines)):
            next_line = lines[i].strip().split(",")
            next_id = next_line[0]

            if cur_id == next_id and i < len(lines) - 1:
                end_idx += 1
            else:
                if i == len(lines) - 1:
                    end_idx += 1
                img = torch.zeros(16, 32, 32, dtype=torch.float32)
                flag = torch.zeros(16, 32, 32, dtype=torch.bool)

                for j in range(start_idx, end_idx):

                    line = lines[j].strip().split(",")

                    t = int(line[2])
                    x = int(float(sensor_info[line[1]]["x"]) - 0.5)
                    y = int(float(sensor_info[line[1]]["y"]) - 0.5)
                    d = 7

                    if j < end_idx - 1:
                        next_line_ = lines[j + 1].strip().split(",")
                        next_t_ = next_line_[2]
                        d = int(next_t_) - t

                    if d <= 0:
                        print("data error: d = {}".format(d))
                        d = 7

                    # todo: check
                    d = (math.log(d) - 2.5)/2.0
                    if d > 0: d /= (4.0 * 1.25)
                    elif d < 0: d /= (1.3 * 1.25)
                    t = (t - 47500.0)/30000

                    for k in range(0, 15, 2):
                        if flag[k, x, y] == False:
                            img[k, x, y] = t
                            img[k+1, x, y] = d
                            flag[k, x, y] = True
                            break

                with open("baselines/GAN/data/data" + str(traj_id) + ".pkl", "wb") as f:
                    pkl.dump(img, f)

                start_idx = end_idx
                cur_id = next_id
                end_idx += 1
                traj_id += 1
                if traj_id % 100 == 0:
                    print(traj_id)

xy_embdID_dict = {}
with open("./data/sensor_info.pkl", "rb") as f:
    sensor_info = pkl.load(f)
    for sid in sensor_info:
        s = sensor_info[sid]
        x = s["x"]
        y = s["y"]
        embdID = s["embedding-id"]
        xy = "{}-{}".format(x, y)
        xy_embdID_dict[xy] = embdID

def restoreTrajFromImg(path_or_tensor, data_type):
    img = None
    if data_type == "path":
        path = path_or_tensor
        with open(path, 'rb') as f:
            img = pkl.load(f)
    elif data_type == "tensor":
        img = path_or_tensor

    z_limit = img.size(dim=0)
    x_limit = img.size(dim=1)
    y_limit = img.size(dim=2)
    seq = []
    for x in range(x_limit):
        for y in range(y_limit):
            tmp = img[:, x, y]
            for z in range(0, 2, z_limit):
                t = tmp[z]
                d = tmp[z + 1]
                if t != 0 and d != 0:
                    if d > 0: d *= (4.0 * 1.25)
                    elif d < 0: d *= (1.3 * 1.25)
                    d = int(math.exp(2 * d.item() + 2.5))
                    t = int(t.item() * 30000 + 47500)
                    try:
                        embdID = xy_embdID_dict["{}-{}".format(x + 0.5, y + 0.5)]
                    except:
                        embdID = -1000
                    seq.append({'x':x, 'y':y, 't':t, 'd':d, 'embedding-id':embdID})
    seq.sort(key=lambda entry : entry['t'])
    return seq
    
with open("./data/key_idx_pair.pkl", "rb") as f:
    key_idx_pair = pkl.load(f)

def extractVisitSeq(result_path_or_seq, arg_type):

    traj = None
    if arg_type == "path":
        result_path = result_path_or_seq
        with open(result_path, "rb") as f:
            traj = pkl.load(f)
    elif arg_type == "seq":
        traj = result_path_or_seq

    visit_seq = []

    for step in traj:
        emb_id = int(step['embedding-id'])
        dur = step['d']
        st_key = getSTKey(dur, emb_id)
        st_idx = None
        if st_key in key_idx_pair:
            st_idx = key_idx_pair[st_key]
            visit_seq.append(st_idx)

    return visit_seq