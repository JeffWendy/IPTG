import pickle as pkl

"""
lift_dict = {
    11410: True,
    11411: True,
    21410: True,
    21411: True,
    10110: True,
    10111: True,
    20111: True,
    20110: True
}

for i in range(1, 4):
    sid_stay = dict()
    stay_points = dict()
    with open("data precessing\Day{}.csv".format(i),encoding = "utf-8") as f:
        head = f.readline()
        step = f.readline()
        traj_dict = dict()
        while step != "":
            tri = step.strip().split(',')
            uid = int(tri[0])
            sid = int(tri[1])
            tstmp = int(tri[2])
            if uid in traj_dict:
                curr_len = traj_dict[uid]["stats"]["len"]
                traj_dict[uid]["steps"].append([sid, tstmp, 0])
                traj_dict[uid]["steps"][curr_len - 1][2] = traj_dict[uid]["steps"][curr_len][1] - traj_dict[uid]["steps"][curr_len-1][1]
                traj_dict[uid]["stats"]["len"] += 1
                dur_pre = traj_dict[uid]["steps"][curr_len - 1][2]
                if dur_pre >= 60:
                    traj_dict[uid]["stats"]["dur_over60"] += 1
                if sid in lift_dict:
                    traj_dict[uid]["stats"]["lift_count"] += 1
            else:
                traj_dict[uid] = {
                    "stats":{
                        "len": 1,
                        "dur_over60": 0,
                        "lift_count": 0
                    },
                    "steps":[[sid, tstmp, 0]]
                }
            step = f.readline()
            
        for uid in traj_dict:
            traj_steps = traj_dict[uid]["steps"]
            for step in traj_steps:
                sid = step[0]
                dur = step[2]
                if sid in sid_stay:
                    sid_stay[sid]["pass_count"] += 1
                    sid_stay[sid]["dur"].append(dur)
                    if dur >= 60:
                        sid_stay[sid]["stay_count"] += 1
                else:
                    sid_stay[sid] = dict()
                    sid_stay[sid]["pass_count"] = 1
                    sid_stay[sid]["dur"] = [dur]
                    if dur >= 60:
                        sid_stay[sid]["stay_count"] = 1
                    else:
                        sid_stay[sid]["stay_count"] = 0

    with open('traj_dict{}.pkl'.format(i), 'wb+') as p:
        pkl.dump(traj_dict, p)

    with open('sid_stay{}.pkl'.format(i), 'wb+') as p:
        pkl.dump(sid_stay, p)
"""

entrance_dict = {
    11300: True,
    11502: True,
    11507: True,
    11504: True,
}

exit_dict = {
    11505: True,
    11515: True,
    11517: True,
    10019: True,
}

"""
for j in range(1, 4):
    # 长度分布；三天和每一天的
    illigal_final_exit_count = 0
    illigal_first_entrance_count = 0
    with open("traj_dict{}.pkl".format(j), "rb") as f:
        traj_dict = pkl.load(f)
    for uid in traj_dict:
        traj = traj_dict[uid]
        traj_len = traj["stats"]["len"]
        steps = traj["steps"]
        traj["stats"]["middle_exit_times"] = 0
        traj["stats"]["exit_count"] = 0
        traj["stats"]["entrance_count"] = 0
        traj["error"] = ""
        for i in range(0, traj_len-1):
            if steps[i][0] in exit_dict:
                traj["stats"]["exit_count"] += 1
                if  steps[i][2] > 300:
                    traj["stats"]["middle_exit_times"] += 1
            if steps[i][0] in entrance_dict:
                traj["stats"]["entrance_count"] += 1
        if  steps[traj_len - 1][0] in exit_dict:
            traj["stats"]["exit_count"] += 1
        else:
            traj["error"] += "illegal final exit;"
            illigal_final_exit_count += 1
        if not steps[0][0] in entrance_dict:
            traj["error"] += "illegal first entrance;"
            illigal_first_entrance_count += 1

    print("{} trajs has an illegal final exit, and {} trajs has an illegal first entrance in Day {}.".format(illigal_final_exit_count, illigal_first_entrance_count, j))
    
    with open("traj_dict{}.pkl".format(j), "wb+") as f:
        pkl.dump(traj_dict, f)
"""

