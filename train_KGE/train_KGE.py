import openke
from openke.config import Trainer, Tester
from openke.module.model import RotatE, ComplEx, TransR, RESCAL
from openke.module.loss import SoftplusLoss, SigmoidLoss, MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./MetaQA/KGE_data/")
parser.add_argument("--EPOCH", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--nbatches", type=int, default=100)
parser.add_argument("--nthreads", type=int, default=8)
parser.add_argument("--bern_flag", type=int, default=1)
parser.add_argument("--filter_flag", type=int, default=1)
parser.add_argument("--neg_ent", type=int, default=25)
parser.add_argument("--neg_rel", type=int, default=0)
parser.add_argument("--sampling_mode", type=str, default="normal")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--model", type=str, default='ComplEx')
parser.add_argument("--embed_dim", type=int, default=256)
parser.add_argument("--cuda", action="store_true", default=False)
parser.add_argument("--alpha", type=int, default=0.1)
parser.add_argument("--regul_rate", type=float, default=0.0)
parser.add_argument("--type_constrain", action="store_true", default=False)
parser.add_argument("--opt_method", type=str, default="adam")
parser.add_argument("--margin", type=float, default=6.0)
parser.add_argument("--epsilon", type=float, default=2.0)
parser.add_argument("--valid_step", type=int, default=10)
parser.add_argument("--save_path", type=str, default="./checkpoint/model_{}.ckpt".format(time.ctime()))

args = parser.parse_args()
train_dataloader = TrainDataLoader(
    in_path=args.data_path,
    nbatches=100,
    threads=args.nthreads,
    sampling_mode=args.sampling_mode,
    bern_flag=args.bern_flag,
    filter_flag=args.filter_flag,
    neg_ent=args.neg_ent,
    neg_rel=args.neg_rel
)

test_dataloader = TestDataLoader(args.data_path, "link")
embed_method = None
if args.model == 'rotateE':
    embed_method = RotatE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.embed_dim,
        margin=args.margin,
        epsilon=args.epsilon
    )
    model = NegativeSampling(
        model=embed_method,
        loss=MarginLoss(margin=4.0),
        batch_size=train_dataloader.get_batch_size(),
        regul_rate=args.regul_rate
    )
elif args.model == 'ComplEx':
    embed_method = ComplEx(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.embed_dim,
    )
    model = NegativeSampling(
        model=embed_method,
        loss=SigmoidLoss(adv_temperature=2),
        batch_size=train_dataloader.get_batch_size(),
        regul_rate=args.regul_rate
    )
elif args.model == 'TransR':
    embed_method = TransR(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=args.embed_dim,
        dim_r=args.embed_dim,
        p_norm=1,
        norm_flag=True,
        rand_init=False
    )
    model = NegativeSampling(
        model=embed_method,
        loss=SigmoidLoss(adv_temperature=2),
        batch_size=train_dataloader.get_batch_size(),
        regul_rate=args.regul_rate
    )
elif args.model == 'RESCAL':
    # python train_KGE.py --cuda --data_path data/MetaQA/ --model RESCAL --opt_method adagrad --bern_flag 1 --filter_flag 1 --neg_ent 25 --sampling_mode normal
    embed_method = RESCAL(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=50
    )

    # define the loss function
    model = NegativeSampling(
        model=embed_method,
        loss=MarginLoss(margin=1.0),
        batch_size=train_dataloader.get_batch_size(),
    )
else:
    raise Exception('unknown embed type')

# train the model
trainer = Trainer(model=model, data_loader=train_dataloader, train_times=args.EPOCH, alpha=args.alpha,
                  use_gpu=args.cuda, opt_method=args.opt_method)
trainer.run()
embed_method.save_checkpoint(args.save_path)

# test the model
embed_method.load_checkpoint(args.save_path)
tester = Tester(model=embed_method, data_loader=test_dataloader, use_gpu=args.cuda)
tester.run_link_prediction(type_constrain=args.type_constrain)