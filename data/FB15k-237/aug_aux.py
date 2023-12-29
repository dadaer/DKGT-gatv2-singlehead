from collections import defaultdict
from tqdm import tqdm
import networkx as nx

id2entity_name = defaultdict(str)
with open('./head10_test/entity2id.txt', 'r') as file:
    entity_lines = file.readlines()
    for line in entity_lines:
        _name, _id = line.strip().split("\t")
        id2entity_name[int(_id)] = _name

id2relation_name = defaultdict(str)

with open('./head10_test/relation2id.txt', 'r') as file:
    relation_lines = file.readlines()
    for line in relation_lines:
        _name, _id = line.strip().split("\t")
        id2relation_name[int(_id)] = _name

unseen_entities = set()
with open('./head10_test/test', 'r') as file:
    test_lines = file.readlines()
    for line in test_lines:
        unseen_entity = line.strip().split("\t")[0]
        unseen_entities.add(unseen_entity)

neighbors = defaultdict(set)
with open('./head10_test/aux', 'r') as file:
    aux_lines = file.readlines()
    for line in aux_lines:
        head, _, tail = line.strip().split("\t")
        if head in unseen_entities:
            neighbors[head].add(tail)
        elif tail in unseen_entities:
            neighbors[tail].add(head)

train_triples = []
with open('./head10_test/train', 'r') as file:
    train_lines = file.readlines()
    for line in train_lines:
        head, rel, tail = line.strip().split("\t")
        train_triples.append([head, rel, tail])

neighbors_inter = defaultdict(list)
for unseen_entity in tqdm(neighbors):
    neis = neighbors[unseen_entity]
    for triple in train_triples:
        head, tail = triple[0], triple[2]
        if head in neis and tail in neis:
            neighbors_inter[unseen_entity].append(triple)

print()

# 最短路径邻接矩阵
# 创建一个图
G = nx.Graph()

# 添加节点和边
for e1, r, e2 in train_triples:
    G.add_node(e1)
    G.add_node(e2)
    G.add_edge(e1, e2)  # 我们可以添加关系作为边的属性，但在计算最短距离时通常不需要它

# 使用Floyd-Warshall算法计算所有节点对之间的最短路径长度
# 这将返回一个节点对字典
lengths = dict(nx.all_pairs_shortest_path_length(G))

# 将最短路径长度转换为邻接矩阵
# 注意：NetworkX通常使用字典来表示图结构，但我们可以转换它为邻接矩阵
nodes = list(G.nodes)
n = len(nodes)
adjacency_matrix = [[float('inf')] * n for _ in range(n)]

# 填充邻接矩阵
for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        if node1 == node2:
            adjacency_matrix[i][j] = 0
        else:
            adjacency_matrix[i][j] = lengths[node1].get(node2, float('inf'))

# 打印邻接矩阵
for row in adjacency_matrix:
    print(row)

# 打印所有实体之间的最短距离
print("所有实体之间的最短距离：")
for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        if i < j:  # 避免重复计算
            print(f"{node1} 到 {node2} 的最短距离是：{adjacency_matrix[i][j]}")

# 一阶关系邻接矩阵
n = len(train_triples)
adjacency_matrix = [[0] * n for _ in range(n)]

# 填充邻接矩阵
for e1_index, r_index, e2_index in train_triples:
    adjacency_matrix[e1_index][e2_index] = r_index
    adjacency_matrix[e2_index][e1_index] = r_index

print()