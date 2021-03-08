import torch
import numpy
from openke.module.model import RotatE
from pytorch_transformers import RobertaModel
BERT_PATH = "C:/Users/yeeeqichen/Desktop/语言模型/roberta-base"


class RelationPredictor(torch.nn.Module):
    def __init__(self):
        super(RelationPredictor, self).__init__()
        print('loading pretrained_model for fine-tuning...')
        self.question_embed = RobertaModel.from_pretrained(BERT_PATH)
        for param in self.question_embed.parameters():
            param.requires_grad = True
        self.hidden2rel = torch.nn.Linear(768, 256)
        torch.nn.init.xavier_uniform_(self.hidden2rel.weight)
        self.classifier = torch.nn.Linear(256, 18)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    # todo:这里直接预测relation的embedding，后续可能会改进
    def forward(self, question_token_ids, question_masks):
        question_embed = self.dropout(
            self.question_embed(input_ids=question_token_ids, attention_mask=question_masks)[1]
        )
        predict_rel = self.hidden2rel(self.relu(question_embed))
        return self.classifier(predict_rel)


class QuestionAnswerModel(torch.nn.Module):
    def __init__(self):
        super(QuestionAnswerModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('using device: {}'.format(self.device))
        self.relation_predictor = RelationPredictor().to(self.device)
        self.KG_embed = RotatE(
            ent_tot=43234,
            rel_tot=18,
            dim=256,
            margin=6.0,
            epsilon=2.0
        )
        self.KG_embed.load_checkpoint('./model_Tue Jan 26 19_24_46 2021.ckpt')
        self.KG_embed.to(self.device)

    def _to_tensor(self, inputs):
        return torch.tensor(inputs).to(self.device)

    # todo: 把这个rotate的计算方式改了,score应为l2范数
    def rotateE(self, head, relation):
        """
        :param head: (batch_size, entity_embed)
        :param relation: (batch_size, relation_embed)
        :return: scores (batch_size, num_entity)
        """
        pi = self.KG_embed.pi_const
        batch_size = head.shape[0]
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(self.KG_embed.ent_embeddings.weight, 2, dim=-1)
        regularized_relation = relation / (self.KG_embed.rel_embedding_range.item() / pi)

        re_relation = torch.cos(regularized_relation)
        im_relation = torch.sin(regularized_relation)
        # (batch_size, ent_tot, entity_embed)
        re_head = re_head.unsqueeze(0).expand(self.KG_embed.ent_tot, -1, -1).permute(1, 0, 2)
        im_head = im_head.unsqueeze(0).expand(self.KG_embed.ent_tot, -1, -1).permute(1, 0, 2)
        re_tail = re_tail.unsqueeze(0).expand(batch_size, -1, -1)
        im_tail = im_tail.unsqueeze(0).expand(batch_size, -1, -1)
        im_relation = im_relation.unsqueeze(0).expand(self.KG_embed.ent_tot, -1, -1).permute(1, 0, 2)
        re_relation = re_relation.unsqueeze(0).expand(self.KG_embed.ent_tot, -1, -1).permute(1, 0, 2)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        # stack: 增加一维对两个tensor进行堆叠，相当于升维
        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        # (batch_size, ent_tot)
        # print(score)
        return score
        # pi = 3.1415926535
        # re_head, im_head = torch.chunk(head, 2, dim=1)
        # re_rotate = torch.cos(relation * pi)
        # im_rotate = torch.sin(relation * pi)
        # real_part = re_head * re_rotate - im_head * im_rotate
        # image_part = re_head * im_rotate + im_head * re_rotate
        # # (num_entities, hidden / 2)
        # re_tail, im_tail = torch.chunk(self.KG_embed.ent_embeddings.weight, 2, dim=1)
        # # (batch, num_entities, hidden / 2)
        # real_part = real_part.abs().unsqueeze(1).repeat(1, self.KG_embed.ent_tot, 1)
        # image_part = image_part.abs().unsqueeze(1).repeat(1, self.KG_embed.ent_tot, 1)
        # # (batch, num_entities)，这里取L1距离进行度量
        # score = (((real_part - re_tail) + (image_part - im_tail)).sum(dim=2)).reciprocal()
        # return score

    def forward(self, question_token_ids, question_masks, head_id):
        rel_scores = self.relation_predictor(self._to_tensor(question_token_ids), self._to_tensor(question_masks))
        print(rel_scores.shape)
        rel_num = torch.max(rel_scores, 1, keepdim=True)
        predict_relation = torch.index_select(self.KG_embed.rel_embeddings.weight, 0, rel_num.indices.view(-1))
        head_embed = self.KG_embed.ent_embeddings(self._to_tensor(head_id)).squeeze(1)
        # print(head_embed.shape, predict_relation.shape)
        scores = self.rotateE(head_embed, predict_relation)
        # print(scores.shape)
        return scores


def test():
    a = QuestionAnswerModel()
    print(a([[1, 2, 3, 0], [1, 1, 1, 0]], [[1, 2, 0, 0], [1, 1, 0, 0]], [[1], [2]]))


if __name__ == '__main__':
    test()

