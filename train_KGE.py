import openke
from openke.config import Trainer, Tester
from openke.module.model import RotatE, ComplEx
from openke.module.loss import SoftplusLoss, SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="WN18")
parser.add_argument("--EPOCH", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--nthreads", type=int, default=8)
parser.add_argument("--bern_flag", type=int, default=0)
parser.add_argument("--filter_flag", type=int, default=1)
parser.add_argument("--neg_ent", type=int, default=64)
parser.add_argument("--neg_rel", type=int, default=0)
parser.add_argument("--sampling_model", type=str, default="cross")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--model", type=str, default='rotateE')
parser.add_argument("--embed_dim", type=int, default=256)
parser.add_argument("--cuda", action="store_true", default=False)
parser.add_argument("--alpha", type=int, default=2e-5)
parser.add_argument("--regul_rate", type=float, default=0.0)
parser.add_argument("--type_constrain", action="store_true", default=False)
parser.add_argument("--opt_method", type=str, default="adam")
parser.add_argument("--margin", type=float, default=6.0)
parser.add_argument("--epsilon", type=float, default=2.0)
parser.add_argument("--valid_step", type=int, default=10)

args = parser.parse_args()
train_dataloader = TrainDataLoader(
    in_path="./benchmarks/" + args.dataset + '/',
    batch_size=args.batch_size,
    threads=args.nthreads,
    sampling_mode=args.sampling_model,
    bern_flag=args.bern_flag,
    filter_flag=args.filter_flag,
    neg_ent=args.neg_ent,
    neg_rel=args.neg_rel
)

test_dataloader = TestDataLoader("./benchmarks/" + args.dataset + '/', "link")
embed_method = None
if args.model == 'rotateE':
    embed_method = RotatE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.embed_dim,
        margin=args.margin,
        epsilon=args.epsilon
    )
elif args.model == 'ComplEx':
    embed_method = ComplEx(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.embed_dim,
    )
else:
    raise Exception('unknown embed type')

model = NegativeSampling(
    model=embed_method,
    loss=SigmoidLoss(adv_temperature=2),
    batch_size=train_dataloader.get_batch_size(),
    regul_rate=args.regul_rate
)

# train the model
trainer = Trainer(model=model, data_loader=train_dataloader, train_times=args.EPOCH, valid_steps=args.valid_step,
                  alpha=args.alpha, use_gpu=args.cuda, opt_method=args.opt_method, test_data_loader=test_dataloader)
trainer.run()
embed_method.save_checkpoint('./checkpoint/simple.ckpt')

# test the model
embed_method.load_checkpoint('./checkpoint/simple.ckpt')
tester = Tester(model=embed_method, data_loader=test_dataloader, use_gpu=args.cuda)
tester.run_link_prediction(type_constrain=args.type_constrain)
