from collections import defaultdict
from tqdm import tqdm

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