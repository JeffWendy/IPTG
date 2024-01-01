import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sb
import pandas as pd


"""
for i in range(1, 4):
    # 长度分布；三天和每一天的
    with open("traj_dict{}.pkl".format(i), "rb") as f:
        traj_dict = pkl.load(f)
    len_dict = dict()
    for uid in traj_dict:
        len = traj_dict[uid]["stats"]["dur_over60"]
        if len in len_dict:
            len_dict[len] += 1
        else:
            len_dict[len] = 1
    len_count_tuples = []
    for l in len_dict:
        len_count_tuples.append((l, len_dict[l]))
    len_count_tuples.sort(key=lambda x: x[0])
    bar_x = [l[0] for l in len_count_tuples]
    bar_y = [l[1] for l in len_count_tuples]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(x=bar_x, height=bar_y)
    ax.set_title("Day{} stay points number distribution".format(i), fontsize=15)
    plt.savefig("Day{} stay points number distribution.png".format(i))
    #plt.show()
"""
"""
for i in range(1, 4):
    # 长度分布；三天和每一天的
    with open("traj_dict{}.pkl".format(i), "rb") as f:
        traj_dict = pkl.load(f)
    len_dict = dict()
    count = 0
    for uid in traj_dict:
        count += 1
        len = traj_dict[uid]["stats"]["len"]
        if len in len_dict:
            len_dict[len] += 1
        else:
            len_dict[len] = 1
    print("Day {} traj count: {}".format(i, count))
    len_count_tuples = []
    for l in len_dict:
        len_count_tuples.append((l, len_dict[l]))
    len_count_tuples.sort(key=lambda x: x[0])
    bar_x = [l[0] for l in len_count_tuples]
    bar_y = [l[1] for l in len_count_tuples]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(x=bar_x, height=bar_y)
    ax.set_title("Day{} traj lengths distribution".format(i), fontsize=15)
    plt.savefig("Day{} traj lengths distribution.png".format(i))
    plt.show()
"""


"""
for j in range(1, 4):
    # 停留频率分布, 阈值60
    with open("sid_stay{}.pkl".format(j), "rb") as f:
        sid_stay = pkl.load(f)
    stay_count_array = np.zeros((33, 30))
    for sid in sid_stay:
        c = sid % 100
        i = int(sid / 100) % 100
        if int(sid / 10000) == 2:
            i += 17 
        stay_count_array[i, c] += sid_stay[sid]["stay_count"]

    idx = [i for i in range(16)] + [-1] + [i for i in range(16)]
    col = [i for i in range(30)]
    data = pd.DataFrame(stay_count_array, index=idx, columns=col)
    plt.figure(dpi=200, figsize=(30,33))
    sb.heatmap(data)
    plt.savefig("Day{} stay points distribution.png".format(j))
    plt.show()
"""
stay_count_array = np.zeros((33, 30))
for j in range(1, 4):
    # 停留频率分布, 阈值180
    with open("data/sid_stay{}.pkl".format(j), "rb") as f:
        sid_stay = pkl.load(f)

    for sid in sid_stay:
        c = sid % 100
        i = int(sid / 100) % 100
        if int(sid / 10000) == 2:
            i += 17 
        for dur in sid_stay[sid]["dur"]:
            if dur > 180:
                stay_count_array[i, c] += 1

idx = [i for i in range(16)] + [-1] + [i for i in range(16)]
col = [i for i in range(30)]
data = pd.DataFrame(stay_count_array, index=idx, columns=col)
plt.figure(dpi=200, figsize=(30,33))
sb.heatmap(data)
plt.savefig("data/Stay points distribution.png")
plt.show()
"""

for j in range(1, 4):
    # 停留点累积时间分布
    with open("sid_stay{}.pkl".format(j), "rb") as f:
        sid_stay = pkl.load(f)
    stay_count_array = np.zeros((33, 30))

    for sid in sid_stay:
        total_dur = 0
        durs = sid_stay[sid]["dur"]
        for dur in durs:
            if dur > 60:
                total_dur += dur
        sid_stay[sid]
        c = sid % 100
        i = int(sid / 100) % 100
        if int(sid / 10000) == 2:
            i += 17 
        stay_count_array[i, c] += sid_stay[sid]["stay_count"]

    idx = [i for i in range(16)] + [-1] + [i for i in range(16)]
    col = [i for i in range(30)]
    data = pd.DataFrame(stay_count_array, index=idx, columns=col)
    plt.figure(dpi=120, figsize=(30,33))
    sb.heatmap(data)
    plt.savefig("Day{} stay points duration distribution.png".format(j))
    plt.show()
    j

"""