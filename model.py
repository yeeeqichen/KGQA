import torch


class KnowledgeEmbeddingModel(torch.nn.Module):
    def __init__(self, entity_size=1, relation_size=1, embed_dim=1, score_func='distMul'):
        super(KnowledgeEmbeddingModel, self).__init__()
        if score_func in ['complEx', 'rotateE']:
            self.embed_dim = 2 * embed_dim
        else:
            self.embed_dim = embed_dim
        self.EntityEmbed = torch.nn.Embedding(entity_size, self.embed_dim)
        torch.nn.init.normal_(self.EntityEmbed.weight.data, mean=0, std=1)
        self.RelationEmbed = torch.nn.Embedding(relation_size, self.embed_dim)
        torch.nn.init.normal_(self.RelationEmbed.weight.data, mean=0, std=1)
        if score_func == 'distMul':
            self.score_func = self.distMul
        elif score_func == 'complEx':
            self.score_func = self.complEx
        else:
            raise Exception('score function error')

    def ce_loss(self, pred, true):
        pred = torch.nn.functional.log_softmax(pred, dim=-1)
        # print(pred.shape, true.shape)
        # true = true / true.size(-1)
        loss = -torch.sum(pred * true)
        return loss

    def distMul(self, head, relation, tail):
        return torch.sum(head * relation * tail, dim=1)

    def complEx(self, head, relation, tail):
        # 目前使用tail=None进行训练
        re_head, im_head = torch.chunk(head, 2, dim=1)
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        if tail is not None:
            re_tail, im_tail = torch.chunk(tail, 2, dim=1)
        else:
            re_tail, im_tail = torch.chunk(self.EntityEmbed.weight, 2, dim=1)
        real_part = re_head * re_relation - im_head * im_relation
        image_part = re_head * im_relation + im_head * re_relation
        if tail is not None:
            re_score = re_tail * real_part - im_tail * image_part
            return torch.sum(re_score, dim=0)  # (batch)
        else:
            re_score = torch.mm(real_part, re_tail.transpose(1, 0)) + torch.mm(image_part, im_tail.transpose(1, 0))
            return re_score  # (batch, entity_num)

    def forward(self, idx1, idx2):
        head_embed = self.EntityEmbed(idx1)
        relation_embed = self.RelationEmbed(idx2)
        score = self.score_func(head_embed, relation_embed, tail=None)
        pred = torch.sigmoid(score)
        return pred



