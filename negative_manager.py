import torch
import random


# todo: negative_samples 的更新策略
class NegativeManager:
    def __init__(self, thresh_hold=15):
        self.thresh_hold = thresh_hold
        self.negative_samples = [[] for _ in range(96106)]

    def step(self, scores, answers, case_ids):
        """
        :param scores: 模型预测的分数
        :param answers: 答案对应的实体id
        :param case_ids: 每一条训练case的id
        :return: None
        """
        # score：(batch_size, answer_size)
        # 这里的indices就是entity id
        values, indices = torch.sort(scores, dim=1, descending=True)
        # 对batch内的每一组分别进行计算
        for entity_ids, case_id, answer in zip(indices, case_ids, answers):
            self.negative_samples[case_id] = []
            for i in range(self.thresh_hold):
                # print(entity_ids[i], answer)
                if int(entity_ids[i]) not in answer:
                    self.negative_samples[case_id].append(int(entity_ids[i]))
            if len(self.negative_samples[case_id]) == 0:
                self.negative_samples[case_id].append(random.randint(0, 43233))
        return values, indices

    def get_negative_samples(self, case_ids):
        negative_samples = []
        for _id in case_ids:
            negative_samples.append(self.negative_samples[_id])
        return negative_samples


def test():
    a = NegativeManager(10)
    print(a.step(torch.randn(2, 10), None, None))


if __name__ == '__main__':
    test()