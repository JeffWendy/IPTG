import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from scipy.special import rel_entr
from algorithms.heap import heapq

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

list_of_pois = [sub_venue_a, sub_venue_b, sub_venue_c, sub_venue_d, 
                register, poster_area, toilet_1, tea_break_1, vip_lounge, 
                lift_1_1, lift_2_1, exhibition_hall, main_venue, service, 
                tea_break_2, media_room, toilet_2, lift_2_2, lift_1_2, toilet_3, 
                work_room, dining_room, hacking_contest_room, relaxation_area]

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

toilet3 = [20, 10, 22, 10]
toilet2 = [14, 27, 14, 29]
toilet1 = [4, 10, 6, 10]
exhibition = [12, 15, 12, 19]
dining_room = [18, 6, 26, 6]
relaxation_area = [29, 6, 31, 6]

special_poi_entrance = [toilet1, toilet2, toilet3,
    exhibition, dining_room, relaxation_area]

lift2_1 = {"1": 1, "2": 1}
lift2_2 = {"338": 1, "339": 1}

lift1_1 = {"303": 1, "304": 1}
lift1_2 = {"462": 1, "463": 1}

list_of_all_places = list_of_pois + list_of_entrances + list_of_exits

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
    if int(minute) < 10: minute = "0" + minute
    if int(second) < 10: second = "0" + second
    return hours + ":" + minute + ":" + second


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

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def read_trajs(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()

    trajs = {}
    max_len = 0

    for line in lines:
        line = line.strip().split(' ')
        length = len(line)
        if length == 1 and (not (length in trajs)):
            length = int(line[0])
            trajs[length] = []
            if max_len < length:
                max_len = length
        elif length > 1:
            traj = [int(i) for i in line]
            trajs[length].append(traj)

    return trajs, max_len

def kl_divergence(dist1, dist2):
    return sum(rel_entr(dist1, dist2))

def normalize_in_place(lst):
    length = len(lst)
    total = sum(lst)
    for i in range(length):
        lst[i] = lst[i] / total
    return lst

def nLargestIndex(n, iterable, key=None):
    return heapq.nlargest(n, iterable, return_index=True)