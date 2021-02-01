import torch


class KnowledgeEmbeddingModel(torch.nn.Module):
    def __init__(self, entity_size=1, relation_size=1, embed_dim=1, use_cuda=False, use_dropout=False,
                 use_batch_norm=False, score_func='distMul', entity_dropout=0.5, relation_dropout=0.5,
                 hidden_dropout=0.5):
        super(KnowledgeEmbeddingModel, self).__init__()
        self.use_cuda = use_cuda
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        if score_func == 'complEx':
            self.entity_embed_dim = 2 * embed_dim
            self.relation_embed_dim = 2 * embed_dim
        elif score_func == 'rotateE':
            self.entity_embed_dim = 2 * embed_dim
            self.relation_embed_dim = embed_dim
        self.EntityEmbed = torch.nn.Embedding(self.entity_size, self.entity_embed_dim)
        torch.nn.init.normal_(self.EntityEmbed.weight.data, mean=0, std=1)
        self.RelationEmbed = torch.nn.Embedding(self.relation_size, self.relation_embed_dim)
        torch.nn.init.normal_(self.RelationEmbed.weight.data, mean=0, std=1)
        if score_func == 'distMul':
            self.score_func = self.distMul
        elif score_func == 'complEx':
            self.score_func = self.complEx
        elif score_func == 'rotateE':
            self.score_func = self.rotateE
        else:
            raise Exception('score function error')
        if self.use_dropout:
            self.entity_dropout = torch.nn.Dropout(entity_dropout)
            self.relation_dropout = torch.nn.Dropout(relation_dropout)
            self.hidden_dropout = torch.nn.Dropout(hidden_dropout)
        if self.use_batch_norm:
            self.batch_norm1 = torch.nn.BatchNorm1d(self.entity_embed_dim)
            self.batch_norm2 = torch.nn.BatchNorm1d(self.relation_embed_dim)
            self.batch_norm3 = torch.nn.BatchNorm1d(2)

    def ce_loss(self, pred, true):
        pred = torch.nn.functional.log_softmax(pred, dim=-1)
        # true = true / true.size(-1)
        loss = -torch.sum(pred * true)
        return loss

    def distMul(self, head, relation, tail):
        return torch.sum(head * relation * tail, dim=1)

    def rotateE(self, head, relation, tail=None):
        # 似乎用这个方法只能采取负采样的办法，否则太慢了
        pi = 3.1415926535
        re_head, im_head = torch.chunk(head, 2, dim=1)
        if tail is not None:
            re_tail, im_tail = torch.chunk(tail, 2, dim=1)
        else:
            re_tail, im_tail = torch.chunk(self.EntityEmbed.weight, 2, dim=1)
        re_rotate = torch.cos(relation * pi)
        im_rotate = torch.sin(relation * pi)
        real_part = re_head * re_rotate - im_head * im_rotate
        image_part = re_head * im_rotate + im_head * re_rotate
        if tail is not None:
            score = None
        else:
            re_tail, im_tail = torch.chunk(self.EntityEmbed.weight, 2, dim=1)  # (num_entities, hidden / 2)
            real_part = real_part.abs().unsqueeze(1).repeat(1, self.entity_size, 1)  # (batch, num_entities, hidden / 2)
            image_part = image_part.abs().unsqueeze(1).repeat(1, self.entity_size, 1)
            # (batch, num_entities)，这里取L1距离进行度量
            score = (((real_part - re_tail) + (image_part - im_tail)).sum(dim=2)).reciprocal()
            # print(score.shape, score)
        return score

    def complEx(self, head, relation, tail=None):
        # 目前使用tail=None进行训练
        # print(head.shape, relation.shape, tail.shape)
        if self.use_batch_norm:
            head = self.batch_norm1(head)
            relation = self.batch_norm2(relation)
        if self.use_dropout:
            head = self.entity_dropout(head)
            relation = self.relation_dropout(relation)
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation, 2, dim=-1)
        if tail is not None:
            re_tail, im_tail = torch.chunk(tail, 2, dim=-1)
        else:
            re_tail, im_tail = torch.chunk(self.EntityEmbed.weight, 2, dim=-1)
        real_part = re_head * re_relation - im_head * im_relation
        image_part = re_head * im_relation + im_head * re_relation
        score = torch.stack([real_part, image_part], dim=1)
        if self.use_batch_norm:
            score = self.batch_norm3(score)
        if self.use_dropout:
            score = self.hidden_dropout(score)
        score = score.permute(1, 0, 2)
        re_score = score[0]
        im_score = score[1]
        if tail is not None:
            re_score = re_tail * real_part - im_tail * image_part
            score = torch.sum(re_score, dim=-1)  # (batch)
        else:
            score = torch.mm(re_score, re_tail.transpose(1, 0)) + torch.mm(im_score, im_tail.transpose(1, 0))
        return score

    def forward(self, head, relation, tail=None):
        head_embed = self.EntityEmbed(head)
        relation_embed = self.RelationEmbed(relation)
        tail_embed = None
        if tail is not None:
            tail_embed = self.EntityEmbed(tail)
        score = self.score_func(head_embed, relation_embed, tail_embed)
        pred = torch.sigmoid(score)
        return pred

    def predict(self, head, relation):
        head_embed = self.EntityEmbed(head)
        relation_embed = self.RelationEmbed(relation)
        score = self.score_func(head_embed, relation_embed)
        pred = torch.sigmoid(score)
        return pred

    def train_step_with_neg(self, positive_sample, negative_sample):
        pos_head = torch.tensor(positive_sample[:, 0])
        pos_rel = torch.tensor(positive_sample[:, 1])
        pos_tail = torch.tensor(positive_sample[:, 2])
        neg_head = torch.tensor(negative_sample[:, :, 0])
        neg_rel = torch.tensor(negative_sample[:, :, 1])
        neg_tail = torch.tensor(negative_sample[:, :, 2])
        if self.use_cuda:
            pos_head = pos_head.cuda()
            pos_rel = pos_rel.cuda()
            pos_tail = pos_tail.cuda()
            neg_head = neg_head.cuda()
            neg_rel = neg_rel.cuda()
            neg_tail = neg_tail.cuda()
        pos_head_embed = self.EntityEmbed(pos_head)
        pos_rel_embed = self.RelationEmbed(pos_rel)
        pos_tail_embed = self.EntityEmbed(pos_tail)
        neg_head_embed = self.EntityEmbed(neg_head)
        neg_rel_embed = self.RelationEmbed(neg_rel)
        neg_tail_embed = self.EntityEmbed(neg_tail)
        pos_score = self.complEx(pos_head_embed, pos_rel_embed, pos_tail_embed)
        neg_score = self.complEx(neg_head_embed, neg_rel_embed, neg_tail_embed)
        # print(pos_score.shape)
        # print(neg_score.shape)
        return pos_score, neg_score
        # pos_score = torch.nn.functional.logsigmoid(pos_score)
        # neg_score = torch.nn.functional.logsigmoid(-neg_score).mean(dim=-1)
        # score = -pos_score - neg_score
        # print(score)
        # return score



