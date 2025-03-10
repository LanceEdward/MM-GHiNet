import numpy as np
from sklearn.model_selection import StratifiedKFold
from node import get_node_feature
import csv
import os
from scipy.spatial import distance

data_folder = "/...."

def standardization_intensity_normalization(dataset, dtype):
    mean = dataset.mean()
    std  = dataset.std()
    return ((dataset - mean) / std).astype(dtype)
def intensityNormalisationFeatureScaling(dataset, dtype):
    max = dataset.max()
    min = dataset.min()

    return ((dataset - min) / (max - min)).astype(dtype)
class dataloader():
    def __init__(self): 
        self.pd_dict = {}
        self.node_ftr_dim = 2000  ##2000
        self.num_classes = 2 

    def load_data(self, connectivity='correlation', atlas='ho'):

        subject_IDs = get_ids()
        labels = get_subject_score(subject_IDs, score='Group')
        num_nodes = len(subject_IDs)
        ages = get_subject_score(subject_IDs, score='Age')
        genders = get_subject_score(subject_IDs, score='Gender')
        #增加基因表达数据
        sites = get_subject_score(subject_IDs, score='SITE_ID')  # 获取SITE_ID 患者脑部点位之间的影响
        sexes = get_subject_score(subject_IDs, score='SEX')  # 获取sexes 患者性别的影响
        ages_at_scan = get_subject_score(subject_IDs, score='AGE_AT_SCAN')  # 获取ages_at_scan 患者年龄的影响
        # apoes = get_subject_score(subject_IDs, score='APOE')
        # clus = get_subject_score(subject_IDs, score='CLU')
        # psen1s = get_subject_score(subject_IDs, score='PSEN1')
        # picalms = get_subject_score(subject_IDs, score='PICALM')
        # trem2s = get_subject_score(subject_IDs, score='TREM2')
        # cr1s = get_subject_score(subject_IDs, score='CR1')
        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])
        age = np.zeros([num_nodes], dtype=np.float32)
        site = np.zeros([num_nodes], dtype=int)  # 初始化SITE_ID
        sex = np.zeros([num_nodes], dtype=int)  # 初始化sex
        gender = np.zeros([num_nodes], dtype=int)
        age_at_scan = np.zeros([num_nodes], dtype=np.float32)  # 初始化age_at_scan
        # apoe = np.zeros([num_nodes], dtype=np.float32)
        # clu = np.zeros([num_nodes], dtype=np.float32)
        # psen1 = np.zeros([num_nodes], dtype=np.float32)
        # picalm = np.zeros([num_nodes], dtype=np.float32)
        # trem2 = np.zeros([num_nodes], dtype=np.float32)
        # cr1 = np.zeros([num_nodes], dtype=np.float32)
        for i in range(num_nodes):
            y_onehot[i, int(labels[subject_IDs[i]])-1] = 1
            y[i] = int(labels[subject_IDs[i]])
            site[i] = int(sites[subject_IDs[i]])
            sex[i] = int(sexes[subject_IDs[i]])
            age_at_scan[i] = float(ages_at_scan[subject_IDs[i]])
            gender[i] = int(genders[subject_IDs[i]])
            # apoe[i] = float(apoes[subject_IDs[i]])
            # clu[i] = float(clus[subject_IDs[i]])
            # psen1[i] = float(psen1s[subject_IDs[i]])
            # picalm[i] = float(picalms[subject_IDs[i]])
            # trem2[i] = float(trem2s[subject_IDs[i]])
            # cr1[i] = float(cr1s[subject_IDs[i]])
        self.y = y
        self.raw_features = get_node_feature()
        phonetic_data = np.zeros([num_nodes, 2], dtype=np.float32)

        phonetic_data[:, 0] = site
        phonetic_data[:, 1] = gender
        # phonetic_data[:, 0] = sex  # 填充sex
        # phonetic_data[:, 1] = age_at_scan  # 填充age_at_scan
        # phonetic_data[:, 2] = apoe

        self.pd_dict['SITE'] = np.copy(phonetic_data[:, 0])
        self.pd_dict['GENDER'] = np.copy(phonetic_data[:, 1])
        # self.pd_dict['SEX'] = np.copy(phonetic_data[:, 0])  # 添加SEX
        # self.pd_dict['AGE_AT_SCAN'] = np.copy(phonetic_data[:, 1])  # 添加AGE_AT_SCAN
        # self.pd_dict['APOE'] = np.copy(phonetic_data[:, 2])  # 添加APOE
        # self.pd_dict['SITE_ID'] = np.copy(phonetic_data[:, 3])  # 添加SITE_ID
        # APOE,CLU,PSEN1,PICALM,TREM2,CR1
        phonetic_score = self.pd_dict
        return self.raw_features, self.y, phonetic_data, phonetic_score

    def data_split(self, n_folds):
        skf = StratifiedKFold(n_splits=n_folds)
        cv_splits = list(skf.split(self.raw_features, self.y))
        return cv_splits

    def get_PAE_inputs(self, nonimg):
 
        n = self.node_ftr.shape[0]
        num_edge = n*(1+n)//2 - n
        pd_ftr_dim = nonimg.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64) 
        edgenet_input = np.zeros([num_edge, 2*pd_ftr_dim], dtype=np.float32)  
        aff_score = np.zeros(num_edge, dtype=np.float32)   
        aff_adj = get_static_affinity_adj(self.node_ftr, self.pd_dict)  
        flatten_ind = 0 
        for i in range(n):
            for j in range(i+1, n):
                edge_index[:,flatten_ind] = [i,j]
                edgenet_input[flatten_ind]  = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j]  
                flatten_ind +=1

        assert flatten_ind == num_edge, "Error in computing edge input"
        
        keep_ind = np.where(aff_score > 1.1)[0]  
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input
    def get_inputs(self, nonimg, embeddings, phonetic_score):

        self.node_ftr = np.array(embeddings)
        n = self.node_ftr.shape[0]
        num_edge = n*(1+n)//2 - n  
        pd_ftr_dim = nonimg.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64) 
        edgenet_input = np.zeros([num_edge, 2*pd_ftr_dim], dtype=np.float32)  
        aff_score = np.zeros(num_edge, dtype=np.float32)   
        aff_adj = get_static_affinity_adj(self.node_ftr, phonetic_score)  
        flatten_ind = 0 
        for i in range(n):
            for j in range(i+1, n):
                edge_index[:,flatten_ind] = [i,j]
                edgenet_input[flatten_ind]  = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j]  
                flatten_ind +=1

        assert flatten_ind == num_edge, "Error in computing edge input"
        
        keep_ind = np.where(aff_score > 1.1)[0]  
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input
def get_subject_score(subject_list, score):
    scores_dict = {}
    # adni ./adni_phenotypic_information.csv
    # abide ./abide_phenotypic_information.csv
    phenotype = "./adni_phenotypic_information.csv"
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['Image Data ID'] in subject_list:
                # 处理空值和缺失值
                scores_dict[row['Image Data ID']] = row[score] if row[score] != '' else '0'

    return scores_dict
def get_ids(num_subjects=None):
    # adni adni_timeseries_subjects_id.txt
    # abide abide_timeseries_subjects_id.txt
    subject_IDs = np.genfromtxt(os.path.join("adni_timeseries_subjects_id.txt"), dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    return subject_IDs

def create_affinity_graph_from_scores(scores, pd_dict, num_nodes):
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = pd_dict[l][:num_nodes]

        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:
                        pass
        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[k] == label_dict[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph

def get_static_affinity_adj(features, pd_dict):
    num_nodes = features.shape[0]
    pd_affinity = create_affinity_graph_from_scores(['SITE','GENDER'], pd_dict, num_nodes)
    distv = distance.pdist(features, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
    adj = pd_affinity * feature_sim
    return adj