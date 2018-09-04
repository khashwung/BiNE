#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'CLH'
# datasketch gives you probabilistic data structures that can process and search very large amount of data
# super fast, with little loss of accuracy
'''
BiNE算法的negative sampling是它的亮点
'''
from datasketch import MinHashLSHForest, MinHash, MinHashLSH
import random

def construct_lsh(obj_dict):
    ''' 构建signature matrix, threshold衡量样本之间哈希值的相似度阈值 '''
    # 阈值的意思是Jaccard similarity > threshold的两个样本被认为是相似的，MinHashLSH用于构建band，以及bucket
    lsh_0 = MinHashLSH(threshold=0, num_perm=128,params=None)
    lsh_5 = MinHashLSH(threshold=0.6, num_perm=128,params=None)
    # forest = MinHashLSHForest(num_perm=128)
    # 所有用户
    keys = obj_dict.keys()
    values = obj_dict.values()
    # ms即为签名矩阵
    ms = []
    for i in range(len(keys)):
        # 128表示签名矩阵有128维，下面temp的更新应该是更新签名矩阵
        temp = MinHash(num_perm=128)
        for d in values[i]:
            # 以下是MinHash的常见用法，因为update的输入时bytes，因此需要encode
            # 可以看到，将某个节点的二阶跳邻居作为当前节点的context，或者当前节点的表达，然后编码成MinHash的哈希表示
            temp.update(d.encode('utf8'))
        # 将该样本添加到签名矩阵中去
        ms.append(temp)
        # 插入当前节点所代表的样本，值为该节点的哈希表示
        lsh_0.insert(keys[i], temp)
        lsh_5.insert(keys[i], temp)
    return lsh_0,lsh_5, keys, ms

def get_negs_by_lsh(user_dict, item_dict, num_negs):
    # 最多采样300个或者以下的负样本，user和item都是
    # 采样率0.01，负样本是正样本的num_negs=4倍
    sample_num_u = max(300, int(len(user_dict)*0.01*num_negs))
    sample_num_v = max(300, int(len(item_dict)*0.01*num_negs))
    negs_u = call_get_negs_by_lsh(sample_num_u,user_dict)
    negs_v = call_get_negs_by_lsh(sample_num_v,item_dict)
    return negs_u,negs_v

def call_get_negs_by_lsh(sample_num, obj_dict):
    lsh_0,lsh_5, keys, ms = construct_lsh(obj_dict)
    visited = []
    negs_dict = {}
    for i in range(len(keys)):
        record = []
        if i in visited:
            continue
        visited.append(i)
        record.append(i)
        total_list = set(keys)
        # query时是用signature对比的
        # 可以看到，lsh_0是用来query相似度>0.0的集合，这个基本上包含了所有的样本，除了相似度=0的样本！
        sim_list = set(lsh_0.query(ms[i]))
        high_sim_list = set(lsh_5.query(ms[i]))
        # 通过集合做差，得到相似度=0的样本集合，这是完全不相似的集合，负样本就从这里采集
        total_list = list(total_list - sim_list)
        for j in high_sim_list:
            total_list = set(total_list)
            # j节点key对应的index
            ind = keys.index(j)
            if ind not in visited:
                visited.append(ind)
                record.append(ind)
            # 对于和i节点最相似的节点j，找到和j至少有那么一点相似的节点集合
            sim_list_child = set(lsh_0.query(ms[ind]))
            # 得到新的total list，这个list是既不和节点i有一点相似，也不和j有一点相似
            total_list = list(total_list - sim_list_child)
        total_list = random.sample(list(total_list), min(sample_num, len(total_list)))
        for j in record:
            key = keys[j]
            # 得到和节点key最不想死的集合
            negs_dict[key] = total_list
    return negs_dict
