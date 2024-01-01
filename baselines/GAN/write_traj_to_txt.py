import os,sys
import pickle as pkl
sys.path.append("./")

from dataProcessing.imgTranslation import restoreTrajFromImg, extractVisitSeq

def restoreTrajsFromImgs(result_dir):
    result = {}

    with open(result_dir, "rb") as f:
        img_batch = pkl.load(f)

    for i in range(img_batch.size(0)):
        traj = restoreTrajFromImg(img_batch[i], "tensor")
        visit_seq = extractVisitSeq(traj, "seq")
        traj_len = len(visit_seq)
        if traj_len in result:
            result[traj_len].append(visit_seq)
        else:
            result[traj_len] = [visit_seq]

    return result

result = restoreTrajsFromImgs('baselines/GAN/results/trajs_epoch180')

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

with open('baselines/GAN/results/trajs_epoch180.txt', 'w') as f:
    for length in result:
        f.write('{}\n'.format(str(length)))
        trajs = result[length]
        for idx, traj in enumerate(trajs):
            traj_txt = traj_to_txt(traj)
            f.write(traj_txt + '\n')