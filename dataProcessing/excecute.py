from dataPreprocessing import *
import multiprocessing as mp

def computeSEDR():

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
    
    #computeAndStoreSubcost(grid_list)

    with open("./data/idx_s_t_tensor.pkl", "rb") as f:
        idx_s_t_tensor = pkl.load(f)

    idx_s_t_list = idx_s_t_tensor.tolist()
    
    with open('real.data') as f:
        lines = f.readlines()

    traj_list = []
    for line in lines:
        line = line.strip()
        steps = line.split(' ')
        length = len(steps)
        if length == 1: 
            continue
        current_traj = []
        traj_list.append(current_traj)
        for step in steps:
            step = int(step)
            embId = idx_s_t_list[step][0]
            x = embId_gridId_map[embId]['x']
            y = embId_gridId_map[embId]['y']
            current_traj.append({'grid-id':embId, 'point':[x, y]})

    traj_count = len(traj_list)
    SEDR_matrix = np.ones((traj_count, traj_count)) * -1
    with open('./data/subcosts.pkl', "rb") as f:
        subcosts = pkl.load(f)

    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores)

    segments = [0, 400, 800, 1200, 1800, 2400, 3400, 4400, 6343]
    # segments = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    results = []
    for i in range(len(segments) - 1):
        result = pool.apply_async(workerProcess, args=(traj_count, segments[i], segments[i+1], traj_list, subcosts))
        results.append(result)
    
    results = [p.get() for p in results]
    SEDR_matrix = np.concatenate(results, axis=0).tolist()

    with open('./data/SEDR_matrix.pkl', 'wb') as f:
        pkl.dump(SEDR_matrix, f)

if __name__ == '__main__':
    computeSEDR()