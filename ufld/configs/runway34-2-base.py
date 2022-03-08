# DATA
dataset = 'CULane'
data_root = '.\\datasets'

# TRAIN
epoch = 101
batch_size = 16
optimizer = 'Adam'  # ['SGD','Adam']
learning_rate = 1e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'  # ['multi', 'cos']
steps = [25, 38]
gamma = 0.1
warmup = 'linear'
warmup_iters = 64

# NETWORK
use_aux = True
griding_num = 32
backbone = '34'

# DATASET
rotate_angel = 30

# LOSS
sim_loss_w = 0
shp_loss_w = 0

# EXP
note = '-34-hd-resize_2-sim0-shp0-r30-aux-no_aug-finetune'
#  hd - resize_2 表示现在HD数据集上训练，再在Resize_2数据集上finetune
log_path = '.\\checkpoints'

# FINETUNE or RESUME MODEL PATH
finetune = '.\\checkpoints\\20220222_124612_lr_1e-02_b_16-34-resize_2-base-sim0-shp0-r30-aux-no_aug\\ep200.pth'
resume = None

# TEST
test_model = '.\\checkpoints\\20220222_143904_lr_1e-04_b_16-34-hd-resize_2-sim0-shp0-r30-aux-no_aug-finetune\\ep100.pth'
test_work_dir = None

num_lanes = 2
