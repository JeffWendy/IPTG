from copy import copy
import math
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dataProcessing.utils import kl_divergence, normalize_in_place, nLargestIndex
from dataProcessing.imgTranslation import restoreTrajFromImg, extractVisitSeq
import pickle as pkl

class NegativeResultTransformer():
    def __init__(self) -> None:
        pass
    
    def loadTrajs(self, negative_sample_path, negative_sample_format=""):
        negative_sample = None
        if negative_sample_format == "trajgene" or negative_sample_format == "seqgan":
            negative_sample = self.__loadTrajsFromTxt(negative_sample_path)
        elif negative_sample_format == "wgan-gp":
            negative_sample = self.__restoreTrajsFromImgs(negative_sample_path)
        elif negative_sample_format == "trajgail":
            negative_sample = self.__restoreTrajsFromTensor(negative_sample_path)
        elif negative_sample_format == 'pkl':
            with open(negative_sample_path, 'rb') as f:
                negative_sample = pkl.load(f)

        return negative_sample

    def __loadTrajsFromTxt(self, negative_sample_dir):
        trajs = {}
        with open(negative_sample_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                steps = line.split(' ')
                if len(steps) == 1: continue
                seq =line.strip().split(' ')
                step_seq = [int(s) for s in seq]
                if len(seq) in trajs:
                    trajs[len(seq)].append(step_seq)
                else:
                    trajs[len(seq)] = [step_seq]
        return trajs


    def __restoreTrajsFromImgs(self, negative_sample_dir):
        negative_sample = {}

        with open(negative_sample_dir, "rb") as f:
            img_batch = pkl.load(f)

        for i in range(img_batch.size(0)):
            traj = restoreTrajFromImg(img_batch[i], "tensor")
            visit_seq = extractVisitSeq(traj, "seq")
            traj_len = len(visit_seq)
            if traj_len in negative_sample:
                negative_sample[traj_len].append(visit_seq)
            else:
                negative_sample[traj_len] = [visit_seq]

        return negative_sample


    def __restoreTrajsFromTensor(self, negative_sample_dir):
        with open(negative_sample_dir, "rb") as f:
            trajs = pkl.load(f)
            
        traj_num = trajs.shape[0]
        max_len = trajs.shape[1]
        negative_sample = {}

        for i in range(traj_num):
            traj_tensor = trajs[i, :]
            traj = []
            for j in range(max_len):
                if traj_tensor[j] == -1:
                    break
                traj.append(traj_tensor[j])
            traj_len = len(traj)
            if traj_len in negative_sample:
                negative_sample[traj_len].append(traj)
            else:
                negative_sample[traj_len] = [traj]  

        return negative_sample
    

class Evaluator():
    def __init__(self, positive_file_dir, interval, topo, n_nodes, labels, max_len, n_clusters) -> None:
        self.trajLoader = NegativeResultTransformer()
        self.topo = topo
        self.labels = labels
        self.n_nodes = n_nodes
        self.interval = interval
        self.n_categories = math.ceil(n_nodes * 1.0 / interval)
        self.n_clusters = n_clusters

        self.positive_distribution = [0.01] * self.n_categories
        self.positive_distribution_among_clusters = {}
        self.positive_len_dist = [0.01 for i in range(max_len + 1)]
        self.positive_len_dist_among_clusters = {}
        self.positive_traj = [list() for i in range(max_len + 1)]
        self.smoothing_func = SmoothingFunction()
        self.negative_distribution = None
        self.negative_distribution_for_clusters = None
        self.negative_length_dist = None

        self.max_len = max_len


        
        self.mapIndicatorToMethod = {
            'node_dist':'node_dist_kl_divergence',
            'top100_dist':'top100_dist_kl_divergence',
            'length_dist':'length_dist_kl_divergence',
            'node_dist_cluster':'node_dist_kl_divergence_cluster',
            'top100_dist_cluster': 'top100_dist_kl_divergence_cluster',
            'topo_score':'topo_violation',
            'bleu_score':'bleu',
        }
        
        traj_count = 0

        with open(positive_file_dir) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            steps = line.split(' ')
            length = len(steps)
            if length == 1: 
                continue
            label = labels[traj_count]
            traj_count += 1

            self.positive_len_dist[length] += 1
            if label not in self.positive_len_dist_among_clusters:
                self.positive_len_dist_among_clusters[label] = [0.01 for i in range(self.max_len + 1)]
            else:
                self.positive_len_dist_among_clusters[label][length] += 1

            if label not in self.positive_distribution_among_clusters:
                self.positive_distribution_among_clusters[label] = [0.01 for i in range(self.n_categories)]

            seq = line.strip().split(' ')
            self.positive_traj[len(seq)].append(seq)
            for step in steps:
                step = int(step)
                if step in [0, 16, 24, 47, 23, 15, 9, 65]:
                    continue

                category = math.floor(step * 1.0 / self.interval)
                self.positive_distribution[category] += 1.0
                self.positive_distribution_among_clusters[label][category] += 1.0

        normalize_in_place(self.positive_len_dist)
        normalize_in_place(self.positive_distribution)

        for label in range(self.n_clusters):
            normalize_in_place(self.positive_distribution_among_clusters[label])
            normalize_in_place(self.positive_len_dist_among_clusters[label])

        self.node_top100_distribution = [0.001] * 100

        self.negative_related_dist_cached = False
        self.negative = None

        self.top100_idx = nLargestIndex(100, self.positive_distribution)
        self.top100_idx_among_clusters = {}
        for label in range(self.n_clusters):
            self.top100_idx_among_clusters[label] = nLargestIndex(100, self.positive_distribution_among_clusters[label])

        self.node_top100_distribution = [self.positive_distribution[self.top100_idx[i]] for i in range(100)]
        for label in range(self.n_clusters):
            tmp = self.positive_distribution_among_clusters[label]
            indexes = self.top100_idx_among_clusters[label]
            self.top100_idx_among_clusters[label] = [tmp[indexes[i]] for i in range(100)]

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __get_negative_node_distribution(self, gene_seqs):
        negative_distribution = [0.01] * self.n_categories
        negative_distribution_for_clusters = {}
        negative_length_dist = [0.01] * (self.max_len + 1)
        step_count = 0.0
        err_count = 0.0

        negative_node_distribution = [0.01 for i in range(self.n_nodes)]

        negative_node_distribution_for_clusters = {}
        for label in range(self.n_clusters):
            negative_node_distribution_for_clusters[label] = [0.01 for i in range(self.n_nodes)]

        for label in range(self.n_clusters):
            negative_distribution_for_cluster = [0.01] * self.n_categories
            gene_seqs_of_label = gene_seqs[label]
            for seq_len in gene_seqs_of_label:
                for seq in gene_seqs_of_label[seq_len]:
                    length = len(seq)
                    negative_length_dist[length] += 1
                    for i, step in enumerate(seq):
                        step_count += 1.0
                        step = int(step)
                        if step in [0, 16, 24, 47, 23, 15, 9, 65]:
                            continue
                        category = math.floor(step * 1.0 / self.interval)
                        negative_distribution[category] += 1.0
                        negative_distribution_for_cluster[category] += 1.0
                        negative_node_distribution[step] += 1.0
                        negative_node_distribution_for_clusters[label][step] += 1.0
                        
                        if i < length - 1:
                            next_step = seq[i + 1]
                            if not self.topo[step, next_step]:
                                err_count += 1.0

            negative_distribution_for_clusters[label] = normalize_in_place(negative_distribution_for_cluster)
            negative_node_distribution_for_clusters[label] = normalize_in_place(negative_node_distribution_for_clusters[label])
        
        self.topo_violation_rate = err_count * 10.0 / step_count
        self.negative_distribution = normalize_in_place(negative_distribution)
        self.negative_distribution_for_clusters = negative_distribution_for_clusters
        self.negative_length_dist = normalize_in_place(negative_length_dist)
        self.negative_node_distribution = normalize_in_place(negative_node_distribution)
        self.negative_node_distribution_for_clusters = negative_node_distribution_for_clusters

        self.negative_distribution_top100 = [0.01] * 100
        self.negative_distribution_top100_for_clusters = {} 
        for label in range(self.n_clusters):
            self.negative_distribution_top100_for_clusters[label] = [0.01] * 100

        for i in range(100):
            nth_idx = self.top100_idx[i]
            self.negative_distribution_top100[i] = self.negative_node_distribution[nth_idx]
            for label in range(self.n_clusters):
                nth_idx = self.top100_idx_among_clusters[label][i]
                self.negative_distribution_top100_for_clusters[label][i] = self.negative_node_distribution_for_clusters[label][i]

        self.negative_distribution_top100 = normalize_in_place(self.negative_distribution_top100)
        self.negative = gene_seqs

    def __compute_and_cache(self, gene_seqs):
        if self.negative_related_dist_cached == False:
            self.__get_negative_node_distribution(gene_seqs)
            self.negative_related_dist_cached = True

    def node_dist_kl_divergence(self, gene_seqs):
        self.__compute_and_cache(gene_seqs)
        return kl_divergence(self.negative_distribution, self.positive_distribution)

    def top100_dist_kl_divergence(self, gene_seqs):
        self.__compute_and_cache(gene_seqs)
        return kl_divergence(self.negative_distribution_top100, self.node_top100_distribution)
    
    def top100_dist_kl_divergence_cluster(self, gene_seqs):
        self.__compute_and_cache(gene_seqs)
        kl_divergence_list = []
        for label in range(self.n_clusters):
            kl_divergence(self.negative_distribution_top100_for_clusters[label], self.top100_idx_among_clusters[label])
            kl_divergence_list.append(kl_divergence(self.negative_distribution_top100_for_clusters[label], self.top100_idx_among_clusters[label]))
        return kl_divergence_list

    def length_dist_kl_divergence(self, gene_seqs):
        self.__compute_and_cache(gene_seqs)
        return kl_divergence(self.negative_length_dist, self.positive_len_dist)

    def node_dist_kl_divergence_cluster(self, gene_seqs):
        # 遍历各个gene_seqs长度        
        self.__compute_and_cache(gene_seqs)
        kl_divergence_list = []
        for label in range(self.n_clusters):            
            kl_divergence_list.append(\
                kl_divergence(self.negative_distribution_for_clusters[label], self.positive_distribution_among_clusters[label]))
        return kl_divergence_list
    
    def topo_violation(self, gene_seqs):
        self.__compute_and_cache(gene_seqs)
        return self.topo_violation_rate
    
    def bleu(self, gene_seqs):
        gene_seq_count = 0.0
        bleu_score = 0.0

        for label in range(self.n_clusters):
            gene_seqs_of_label = gene_seqs[label]
            for seq_len in gene_seqs_of_label:
                if seq_len >= 35:
                    continue
                candidate_seqs = gene_seqs_of_label[seq_len]
                candidate_num = len(candidate_seqs)
                if candidate_num > 5:
                    candidate_seqs = candidate_seqs[0 : math.floor(candidate_num * 1.0 / 5)]
                reference_seqs = copy(self.positive_traj[seq_len])
                #random.shuffle(reference_seqs)
                if seq_len > 3:
                    reference_seqs += self.positive_traj[seq_len - 1]
                    reference_seqs += self.positive_traj[seq_len - 2]
                if seq_len < 32:
                    reference_seqs += self.positive_traj[seq_len + 1]
                    reference_seqs += self.positive_traj[seq_len + 2]
                for candidate_seq in candidate_seqs:
                    candidate_seq = [str(i) for i in candidate_seq]
                    if candidate_seq[0] == '2751' and candidate_seq[-1] == '2752':
                        candidate_seq = candidate_seq[1:-1]
                    if candidate_seq[0] == '2751' and candidate_seq[-1] != '2752':
                        candidate_seq = candidate_seq[1:]
                    if candidate_seq[0] != '2751' and candidate_seq[-1] == '2752':
                        candidate_seq = candidate_seq[:-1]
                    gene_seq_count += 1.0
                    bleu_score += sentence_bleu(reference_seqs, candidate_seq, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing_func.method1)
        
        bleu_score /= gene_seq_count * 1.0
        return bleu_score

    def self_bleu(self):
        gene_seq_count = 0.0
        bleu_score = 0.0
        for trajs in self.positive_traj:
            if len(trajs) == 0:
                continue
            seq_len = len(trajs[0])
            candidate_seqs = trajs
            candidate_num = len(candidate_seqs)
            if candidate_num > 10:
                candidate_seqs = candidate_seqs[0 : math.floor(candidate_num * 1.0 / 5)]
                candidate_num = len(candidate_seqs)
            reference_seqs = []
            random.shuffle(reference_seqs)
            if seq_len > 3:
                reference_seqs += self.positive_traj[seq_len - 1]
                reference_seqs += self.positive_traj[seq_len - 2]
            if seq_len < 33:
                reference_seqs += self.positive_traj[seq_len + 1]
                reference_seqs += self.positive_traj[seq_len + 2]
            for idx, candidate_seq in enumerate(candidate_seqs):
                tmp = None
                if idx == 0:
                    tmp = reference_seqs + self.positive_traj[seq_len][1:candidate_num]
                elif idx == candidate_num - 1:
                    tmp = reference_seqs + self.positive_traj[seq_len][0:candidate_num-1]
                else:
                    tmp = reference_seqs + self.positive_traj[seq_len][0:idx] + self.positive_traj[seq_len][idx+1:candidate_num]
                candidate_seq = [str(i) for i in candidate_seq]
                tmp = sentence_bleu(tmp, candidate_seq, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing_func.method1)
                bleu_score += tmp
                gene_seq_count += 1.0
        bleu_score /= gene_seq_count * 1.0
        return bleu_score
    
    def evaluate(self, negative_sample_paths=[], negative_sample_formats=[], metrics=[]) -> list:
        if isinstance(negative_sample_formats, str) and isinstance(negative_sample_paths, str):
            return self.__loadAndEvaluate(negative_sample_paths, negative_sample_formats, metrics)
        
        if len(negative_sample_paths) != len(negative_sample_formats):
            raise Exception('len(negative_sample_paths) != len(negative_sample_formats)')
        
        results = []
        for i in range(negative_sample_paths):
            negative_sample_path = negative_sample_paths[i]
            negative_sample_format = negative_sample_formats[i]
            results.append(self.__loadAndEvaluate(negative_sample_path, negative_sample_format, metrics))
        
        return results

    def __loadAndEvaluate(self, negative_sample_path, negative_sample_format, metrics):
        negative_sample = self.trajLoader.loadTrajs(negative_sample_path, negative_sample_format)
        if isinstance(metrics, str):
            method_name = self.__map_metric_to_method(metrics)
            return self[method_name](negative_sample)
        else:
            results = []
            for metric in metrics:
                method_name = self.__map_metric_to_method(metric)
                results.append(self[method_name](negative_sample))
        return results
    
    def __map_metric_to_method(self, metric):
        if metric in self.mapIndicatorToMethod:
            method_name = self.mapIndicatorToMethod[metric]
            return method_name
        else:
            raise Exception('metric {} not supported!'.format(metric))