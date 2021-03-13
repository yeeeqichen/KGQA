from dataloader import DataLoader
from QA_model import QuestionAnswerModel
import torch
import argparse
import logging
import tqdm
from pytorch_transformers import AdamW
import os
import time

parser = argparse.ArgumentParser()

parser.add_argument("--train_file", type=str, default="./MetaQA/QA_data/qa_train_1hop.txt")
parser.add_argument("--valid_file", type=str, default="./MetaQA/QA_data/qa_dev_1hop.txt")
parser.add_argument("--test_file", type=str, default="./MetaQA/QA_data/qa_test_1hop.txt")
parser.add_argument("--dict_path", type=str, default="./MetaQA/QA_data/entities.dict")
parser.add_argument("--relation_file", type=str, default='./MetaQA/KGE_data/relation2id.txt')
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--seq_length", type=int, default=20)
parser.add_argument("--EPOCH", type=int, default=10)
parser.add_argument("--valid_steps", type=int, default=1000)
parser.add_argument("--log_level", type=str, default="DEBUG")
parser.add_argument("--require_improvement", type=int, default=100)
parser.add_argument("--save_path", type=str, default='/model/' + time.strftime("%Y-%m-%d__%H-%M-%S", time.localtime()))
parser.add_argument("--require_save", action='store_true', default=True)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--adam_epsilon", default=1e-8, type=float)
parser.add_argument("--max_gradient_norm", type=int, default=10)
parser.add_argument("--scheduler_steps", type=int, default=100)


args = parser.parse_args()

save_path = os.getcwd() + args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger = logging.getLogger(__name__)
logger.setLevel(args.log_level)
formatter = logging.Formatter('%(asctime)s -- %(levelname)s - %(name)s - %(message)s')

sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

fh = logging.FileHandler(save_path + '/log.txt')
fh.setFormatter(formatter)
logger.addHandler(fh)


# todo: 增加训练时参数的保存（done）；优化relation预测（done，改为softmax）；在loss中尝试加入negative项
def train(model, data_loader):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.95, patience=0, verbose=True)
    best_performance = {'hits_1': 0, 'hits_3': 0, 'hits_10': 0}
    require_improvement = 0
    total_loss = []
    model.train()
    for i in range(args.EPOCH):
        steps = 1
        for question_token_ids, question_masks, head_id, answers_id in tqdm.tqdm(data_loader.batch_generator('train')):
            model.zero_grad()
            scores = model(question_token_ids, question_masks, head_id).cpu()
            cur_loss = []
            for score, answers in zip(scores, answers_id):
                target_scores = torch.index_select(score, 0, torch.tensor(answers))
                cur_loss.append(torch.sum(-torch.log(torch.sigmoid(target_scores))))
                # loss.append(torch.sum(target_scores))
                # loss.append(torch.sum(-torch.log(target_scores)))
            train_loss = torch.stack(cur_loss).mean()
            total_loss.append(train_loss)
            if steps % args.scheduler_steps == 0:
                average_loss = torch.stack(total_loss).mean()
                logger.info('EPOCH: {}, STEP: {}, average loss: {}'.format(i, steps, average_loss))
                total_loss = []
            train_loss.backward()
            # 进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_gradient_norm)
            optimizer.step()
            if steps % args.valid_steps == 0:
                logger.info('Start validating...')
                hits_1 = 0
                hits_3 = 0
                hits_10 = 0
                cnt = 0
                for _question, _mask, _head, _answers in data_loader.batch_generator('valid'):
                    scores = model(_question, _mask, _head).cpu()
                    predicts = torch.sort(scores, dim=1, descending=True).indices
                    for predict, _answer in zip(predicts, _answers):
                        cnt += 1
                        for j in range(10):
                            if predict[j] in _answer:
                                if j == 0:
                                    hits_1 += 1
                                if j < 3:
                                    hits_3 += 1
                                if j < 10:
                                    hits_10 += 1
                                break
                cur_performance = {'hits_1': hits_1 / cnt, 'hits_3': hits_3 / cnt, 'hits_10': hits_10 / cnt}
                logger.info('EPOCH: {}, STEP: {}, Hits_1: {}, Hits_3: {}, Hits_10: {}'
                            .format(i, steps, cur_performance['hits_1'], cur_performance['hits_3'],
                                    cur_performance['hits_10']))
                # 依据验证集上的表现来调整学习率
                scheduler.step(cur_performance['hits_1'])
                if cur_performance['hits_1'] > best_performance['hits_1']:
                    best_performance = cur_performance
                    if args.require_save:
                        logger.info('Saving model...')
                        torch.save(model.state_dict(), save_path + '/model.pkl')
                        with open(save_path + '/performance.txt', 'w') as f:
                            f.write("Best Performance, Hits_1: {}, Hits_3: {}, Hist_10: {}".
                                    format(best_performance['hits_1'], best_performance['hits_3'],
                                           best_performance['hits_10']))
                        with open(save_path + '/config.txt', 'w') as f:
                            for eachArg, value in args.__dict__.items():
                                f.writelines(eachArg + ' : ' + str(value) + '\n')
                    require_improvement = 0
                else:
                    require_improvement += 1
                    if require_improvement == args.require_improvement:
                        logger.warning('EXIT: training finished because of no improvement')
                        exit(-1)
            steps += 1
    logger.info('finish training')


def main():
    model = QuestionAnswerModel(embed_model_path='./model_Tue Jan 26 19_24_46 2021.ckpt')
    total_param = 0
    for name, param in model.named_parameters():
        num = 1
        for size in param.shape:
            num *= size
        total_param += num
        logger.info("{:30s} : {}, require_grad: {}".format(name, param.shape, param.requires_grad))
    logger.info("total param num {}".format(total_param))
    data_loader = DataLoader(
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        dict_path=args.dict_path
    )
    model.to(model.device)
    train(model, data_loader)


if __name__ == '__main__':
    main()
