from py2neo import Graph, Node, Relationship, NodeMatcher

graph = Graph("bolt://localhost:7687", auth=('neo4j', 'neo4j'))
# graph = Graph(
#     "http://localhost:7474",
#     username="neo4j",
#     password="neo4j"
# )
matcher = NodeMatcher(graph)
ent_dict = {}
rel_dict = {}
with open('MetaQA/KGE_data/entity2id.txt') as f:
    for line in f:
        tokens = line.strip('\n').split('\t')
        if len(tokens) < 2:
            continue
        ent_dict[tokens[1]] = tokens[0]
        # graph.create(Node('None', name=tokens[0]))
with open('MetaQA/KGE_data/relation2id.txt') as f:
    for line in f:
        tokens = line.strip('\n').split('\t')
        if len(tokens) < 2:
            continue
        rel_dict[tokens[1]] = tokens[0]

with open('MetaQA/KGE_data/train2id.txt') as f:
    for line in f:
        tokens = line.strip('\n').split(' ')
        if len(tokens) < 3:
            continue
        head_name = ent_dict[tokens[0]]
        tail_name = ent_dict[tokens[1]]
        rel_name = rel_dict[tokens[2]]
        print(head_name, tail_name, rel_name)
        graph.create(Relationship(matcher.match('None', name=head_name).first(), rel_name,
                                  matcher.match('None', name=tail_name).first()))


