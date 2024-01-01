import pickle as pkl

with open('baselines/TrajGAIL/results/result800', "rb") as f:
    trajs = pkl.load(f)
    
traj_num = trajs.shape[0]
max_len = trajs.shape[1]
result = {}

for i in range(traj_num):
    traj_tensor = trajs[i, :]
    traj = []
    for j in range(max_len):
        if traj_tensor[j] == -1:
            break
        traj.append(traj_tensor[j])
    traj_len = len(traj)
    if traj_len in result:
        result[traj_len].append(traj)
    else:
        result[traj_len] = [traj]  

def traj_to_txt(li):
    s = ''
    length = len(li)
    for idx, i in enumerate(li):
        if idx == 0:
            continue
        if idx == length -1:
            continue
        s += str(i) + ' '
    s.strip()
    return s

with open('baselines/TrajGAIL/results/result800.txt', 'w') as f:
    for length in result:
        f.write('{}\n'.format(str(length)))
        trajs = result[length]
        for idx, traj in enumerate(trajs):
            traj_txt = traj_to_txt(traj)
            f.write(traj_txt + '\n')