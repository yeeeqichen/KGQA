import torch
import logging
from openke.module.model import RotatE, ComplEx
from pytorch_transformers import RobertaModel
from sklearn.cluster import KMeans
import numpy as np
logger = logging.getLogger(__name__)


class CandidateGenerator:
    def __init__(self, train2id):
        self.candidate_entities = [[] for _ in range(43234)]
        with open(train2id) as f:
            for _, line in enumerate(f):
                if _ == 0:
                    continue
                head, tail, relation = line.strip('\n').split(' ')
                self.candidate_entities[int(head)].append((int(tail)))
                self.candidate_entities[int(tail)].append((int(head)))
        for i in range(43234):
            self.candidate_entities[i] = sorted(list(set(self.candidate_entities[i])))
        self.candidate_entities = np.array(self.candidate_entities)

    def get_candidates(self, entity_id):
        return self.candidate_entities[entity_id]


class RelationPredictor(torch.nn.Module):
    def __init__(self, bert_path, fine_tune=True, attention=True):
        super(RelationPredictor, self).__init__()
        self.relation_names = []
        with open('./MetaQA/KGE_data/relation2id.txt') as f:
            for _, line in enumerate(f):
                if _ == 0:
                    continue
                relation, _id = line.split('\t')
                self.relation_names.append(relation.replace('_', ' '))
        # 这个矩阵存储了每个entity可以关联的relation的类型，用于后续relation的预测
        self.adjacencyMatrix = [[0.] * 18 for _ in range(43234)]
        with open('./MetaQA/KGE_data/train2id.txt') as f:
            for _, line in enumerate(f):
                if _ == 0:
                    continue
                head, tail, relation = line.strip('\n').split(' ')
                self.adjacencyMatrix[int(head)][int(relation)] = 1.
        self.adjacencyHandler = torch.nn.Linear(18, 18)
        # print(self.adjacencyMatrix[3])
        # print(self.adjacencyMatrix[333])
        self.fine_tune = fine_tune
        self.attention = attention
        logger.info('loading pretrained bert model...')
        self.question_embed = RobertaModel.from_pretrained(bert_path)
        if self.fine_tune:
            for param in self.question_embed.parameters():
                param.requires_grad = True
            self.hidden2rel = torch.nn.Linear(768, 18)
            torch.nn.init.xavier_uniform_(self.hidden2rel.weight)
        elif self.attention:
            for param in self.question_embed.parameters():
                param.requires_grad = False
            # self.attention_w = torch.nn.Linear(768, 1)
            # self.tanh = torch.nn.Tanh()
            self.hidden2rel = torch.nn.Sequential(
                torch.nn.Linear(768, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 18)
            )
            # self.hidden2rel = torch.nn.Linear(768, 18)
            self.attention_key = torch.nn.Linear(768, 768)  # hello my name is yeeeeeqichen
            self.attention_query = torch.nn.Linear(768, 768)  # I am a student from Peking Univ
            self.attention_value = torch.nn.Linear(768, 768)  # and I like coding!!!!!
            self.feed_forward = torch.nn.Linear(768, 768)
            self.attention_scores = []
        else:
            for param in self.question_embed.parameters():
                param.requires_grad = False
            self.hidden2rel = torch.nn.Sequential(
                torch.nn.Linear(768, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 18)
            )
        # self.classifier = torch.nn.Linear(768, 18)
        # torch.nn.init.normal_(self.classifier.weight, mean=0, std=1)
        # logger.info(self.classifier.weight)
        # self.relu = torch.nn.ReLU()
        # self.dropout = torch.nn.Dropout(0.5)

    def forward(self, question_token_ids, question_masks):
        if not self.attention:
            question_embed = torch.mean(self.question_embed(input_ids=question_token_ids,
                                                            attention_mask=question_masks)[0], dim=1)

            # # [CLS]
            # last_hidden_state = self.question_embed(input_ids=question_token_ids, attention_mask=question_masks)[0]
            # question_embed = last_hidden_state.transpose(1, 0)[0]
            # classification
        else:
            # (batch_size, sequence_length, hidden_size)
            last_hidden_states = self.question_embed(input_ids=question_token_ids, attention_mask=question_masks)[0]
            # (batch_size, sequence_length, 1)
            # attention_weight = torch.softmax(self.tanh(self.attention_w(last_hidden_states)), dim=1)
            # question_embed = torch.sum(last_hidden_states * attention_weight, dim=1)
            query = self.attention_query(last_hidden_states)
            key = self.attention_key(last_hidden_states)
            value = self.attention_value(last_hidden_states)
            # (batch_size, sequence_length, sequence_length)
            energy = torch.matmul(query, key.transpose(-1, -2))
            # print(question_masks)
            attention_mask = (1.0 - question_masks.unsqueeze(1)) * -10000.0
            # print(attention_mask)
            prob = torch.softmax(energy + attention_mask, dim=-1)
            # print(prob)
            attention_output = torch.matmul(prob, value)

            feed_forward_output = self.feed_forward(attention_output)

            question_embed = torch.mean(attention_output + feed_forward_output, dim=1)

            # self.attention_scores.append([question_token_ids, attention_weight])

            # if len(self.attention_scores) > 1000:
            #     print(self.attention_scores[-1])
            #     self.attention_scores = []

        predict_rel = self.hidden2rel(question_embed)
        return predict_rel


class QuestionAnswerModel(torch.nn.Module):
    def __init__(self, embed_model_path, bert_path, n_clusters, embed_method='rotatE', fine_tune=True, attention=True):
        super(QuestionAnswerModel, self).__init__()
        self.embed_method = embed_method
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info('using device: {}'.format(self.device))
        self.relation_predictor = RelationPredictor(bert_path=bert_path, fine_tune=fine_tune,
                                                    attention=attention).to(self.device)
        if self.embed_method == 'rotatE':
            self.score_func = self.rotatE
            self.KG_embed = RotatE(
                ent_tot=43234,
                rel_tot=18,
                dim=256,
                margin=6.0,
                epsilon=2.0
            )
        elif self.embed_method == 'complEx':
            self.score_func = self.complEx
            self.KG_embed = ComplEx(
                ent_tot=43234,
                rel_tot=18,
                dim=256
            )
        else:
            exit(1)
        self.embed_model_path = embed_model_path
        self.KG_embed.load_checkpoint(self.embed_model_path)
        self.KG_embed.to(self.device)
        for param in self.KG_embed.parameters():
            param.requires_grad = False
        logger.info('loading pretrained KG embedding from {}'.format(self.embed_model_path))
        if self.embed_method == 'rotatE':
            self.cluster = KMeans(n_clusters=n_clusters)
            self.cluster2ent = [[] for _ in range(n_clusters)]
            for idx, label in enumerate(self.cluster.fit_predict(self.KG_embed.ent_embeddings.weight.cpu())):
                self.cluster2ent[label].append(idx)
        self.candidate_generator = CandidateGenerator('./MetaQA/KGE_data/train2id.txt')
        # cnt = 0
        # for _ in self.cluster2ent:
        #     cnt += len(_)
        # assert cnt == self.KG_embed.ent_tot

    def _to_tensor(self, inputs):
        return torch.tensor(inputs).to(self.device)

    def complEx(self, head, relation, tail):
        """
        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1
        )
        :param head:
        :param relation:
        :param tail:
        :return:
        """
        batch_size = head.shape[1]
        target_size = tail.shape[1]
        # print(batch_size, target_size)
        re_head, im_head = torch.chunk(head.squeeze(2), 2, dim=0)
        re_tail, im_tail = torch.chunk(tail.squeeze(2), 2, dim=0)
        re_relation, im_relation = torch.chunk(relation.squeeze(2), 2, dim=0)
        # 统一转换成(batch_size, target_size, embed_size)
        # print(re_head.shape, re_tail.shape, re_relation.shape)
        re_head = re_head.expand(target_size, -1, -1).permute(1, 0, 2)
        im_head = im_head.expand(target_size, -1, -1).permute(1, 0, 2)
        re_tail = re_tail.expand(batch_size, -1, -1)
        im_tail = im_tail.expand(batch_size, -1, -1)
        im_relation = im_relation.expand(target_size, -1, -1).permute(1, 0, 2)
        re_relation = re_relation.expand(target_size, -1, -1).permute(1, 0, 2)

        score = torch.sum(re_head * re_tail * re_relation + im_head * im_tail * re_relation +
                          re_head * im_tail * im_relation - im_head * re_tail * im_relation, -1)
        # (batch_size, target_size)
        # print(score.shape)
        return score

    def rotatE(self, head, relation, tail):
        """
        :param head: (batch_size, entity_embed)
        :param relation: (batch_size, relation_embed)
        :param tail: (target_size, entity_embed)
        :return: scores (batch_size, num_entity)
        """
        pi = self.KG_embed.pi_const
        batch_size = head.shape[0]
        target_size = tail.shape[0]
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)
        regularized_relation = relation / (self.KG_embed.rel_embedding_range.item() / pi)

        re_relation = torch.cos(regularized_relation)
        im_relation = torch.sin(regularized_relation)
        # (batch_size, ent_tot, entity_embed)
        re_head = re_head.unsqueeze(0).expand(target_size, -1, -1).permute(1, 0, 2)
        im_head = im_head.unsqueeze(0).expand(target_size, -1, -1).permute(1, 0, 2)
        re_tail = re_tail.unsqueeze(0).expand(batch_size, -1, -1)
        im_tail = im_tail.unsqueeze(0).expand(batch_size, -1, -1)
        im_relation = im_relation.unsqueeze(0).expand(target_size, -1, -1).permute(1, 0, 2)
        re_relation = re_relation.unsqueeze(0).expand(target_size, -1, -1).permute(1, 0, 2)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        # stack: 增加一维对两个tensor进行堆叠，相当于升维
        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        # (batch_size, ent_tot)
        return self.KG_embed.margin - score

    # 经实验 sigmoid效果最好
    def forward(self, question_token_ids, question_masks, head_id, use_cluster=False):
        rel_scores = self.relation_predictor(self._to_tensor(question_token_ids), self._to_tensor(question_masks))
        _index = [_[0] for _ in head_id]
        # print(_index)
        adjacency_scores = torch.index_select(self._to_tensor(self.relation_predictor.adjacencyMatrix), 0,
                                              self._to_tensor(_index))
        adjacency_scores = self.relation_predictor.adjacencyHandler(adjacency_scores)
        rel_scores = (rel_scores + adjacency_scores) / 2
        # print(adjacency_scores)
        # relation的预测方式采用self.KG_embed.rel_embeddings.weight的线性组合，取sigmoid（scores）作为组合系数

        # print(predict_relation)
        # predict_relation = torch.clip(predict_relation,
        #                               min=-self.KG_embed.rel_embedding_range.weight.data,
        #                               max=self.KG_embed.rel_embedding_range.weight.data)
        # print(predict_relation)
        # predict_relation = self.relation_predictor(self._to_tensor(question_token_ids),
        # self._to_tensor(question_masks))
        if self.embed_method == 'complEx':
            _tensor = self._to_tensor(head_id)
            head_embed = torch.stack([self.KG_embed.ent_re_embeddings(_tensor),
                                      self.KG_embed.ent_im_embeddings(_tensor)], dim=0)
            predict_relation = torch.matmul(torch.sigmoid(rel_scores), torch.stack(
                [self.KG_embed.rel_re_embeddings.weight, self.KG_embed.rel_im_embeddings.weight], dim=0))

        else:
            head_embed = self.KG_embed.ent_embeddings(self._to_tensor(head_id)).squeeze(1)
            predict_relation = torch.matmul(torch.sigmoid(rel_scores), self.KG_embed.rel_embeddings.weight)

        # candidate_answers = list(self.candidate_generator.get_candidates(_index))
        # print(_index, candidate_answers)
        indices = None
        if not use_cluster:
            if self.embed_method == 'complEx':
                tail_embed = torch.stack([self.KG_embed.ent_re_embeddings.weight,
                                          self.KG_embed.ent_im_embeddings.weight], dim=0)
            else:
                tail_embed = self.KG_embed.ent_embeddings.weight
            # scores越大越好
            scores = self.score_func(head_embed, predict_relation, tail_embed)
            return scores
        else:
            centers = self.cluster.cluster_centers_
            cluster_scores = self.score_func(head_embed, predict_relation, self._to_tensor(centers))
            # print(cluster_scores)
            values, indices = torch.max(cluster_scores, dim=1)
            # print(values, indices)
            tail_embed = []
            for cluster_index in indices:
                tail_embed.append(torch.index_select(self.KG_embed.ent_embeddings.weight, 0,
                                                     self._to_tensor(self.cluster2ent[cluster_index])))
            scores = []
            for _head, _rel, _tail in zip(head_embed, predict_relation, tail_embed):
                scores.append(self.score_func(_head.unsqueeze(0), _rel.unsqueeze(0), _tail))
            # print(scores)
            return scores, indices


def test():
    model = QuestionAnswerModel(embed_model_path='checkpoint/rotatE.ckpt',
                            bert_path="C:/Users/yeeeqichen/Desktop/语言模型/roberta-base", n_clusters=8, fine_tune=False)
    total_param = 0
    for name, param in model.named_parameters():
        num = 1
        for size in param.shape:
            num *= size
        total_param += num
        print("{:30s} : {}, require_grad: {}".format(name, param.shape, param.requires_grad))
    print("total param num {}".format(total_param))
    print(model([[1, 2, 3, 0], [1, 4, 0, 0]], [[1, 1, 1, 0], [1, 1, 0, 0]], [[1], [2]]))


if __name__ == '__main__':
    test()

