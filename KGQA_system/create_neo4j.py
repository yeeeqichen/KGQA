from py2neo import Graph, Node, Relationship, NodeMatcher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="MetaQA/KGE_data/")
parser.add_argument("--name", type=str, default="neo4j", help="neo4j数据库登陆名称")
parser.add_argument("--password", type=str, default="neo4j", help="neo4j数据库登陆密码")

args = parser.parse_args()

graph = Graph("bolt://localhost:7687", auth=(args.name, args.password))
# graph = Graph(
#     "http://localhost:7474",
#     username="neo4j",
#     password="neo4j"
# )
matcher = NodeMatcher(graph)
ent_dict = {}
rel_dict = {}
with open(args.data_path + 'entity2id.txt') as f:
    for line in f:
        tokens = line.strip('\n').split('\t')
        if len(tokens) < 2:
            continue
        ent_dict[tokens[1]] = tokens[0]
        # graph.create(Node('None', name=tokens[0]))
with open(args.data_path + 'relation2id.txt') as f:
    for line in f:
        tokens = line.strip('\n').split('\t')
        if len(tokens) < 2:
            continue
        rel_dict[tokens[1]] = tokens[0]

with open(args.data_path + 'train2id.txt') as f:
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


