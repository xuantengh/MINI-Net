
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='choose a dataset')
parser.add_argument('--domain', type=str, help='choose a domain')
# parser.add_argument('--stddev', type=float, default=.1)
parser.add_argument('--score_path', type=str)
parser.add_argument('--GT_path', type=str)
#optimizer
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--lr_decay', type=float, default=0.7)
parser.add_argument('--decay_epoch', type=float, default=20)


#constant
parser.add_argument('--num_per_group', type=int, default=8)
parser.add_argument('--feature_dim', type=int, default=512)
parser.add_argument('--audio_dim', type=int, default=128)

parser.add_argument('--frames_per_clip', type=int, default=30)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--short_lower',type=int,default = 4)
parser.add_argument('--short_upper',type=int,default = 20)
parser.add_argument('--long_lower',type=int,default = 40)
parser.add_argument('--long_upper',type=int,default = 60)
parser.add_argument('--interval',type=int,default=10)
parser.add_argument('--topk',type=int,default=4)

parser.add_argument('--FNet',type=str)
parser.add_argument('--HNet',type=str)
parser.add_argument('--AM',type=str)
parser.add_argument('--DS',type=str)
parser.add_argument('--AHLoss',type=str)


#input
parser.add_argument('--temporalN', type=int,default=5)

parser.add_argument('--test_path', type=str)
parser.add_argument('--train_path', type=str)
parser.add_argument('--topk_mAP',type = int, default=1)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--alpha', type=float, default=0.7)
parser.add_argument('--paper_data_split',action='store_true')
parser.add_argument('--bagsize',type=int, default=60)


args = parser.parse_args()
