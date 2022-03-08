# DATA
dataset = 'CULane'
data_root = '.\\datasets'

# TRAIN
epoch = 201
batch_size = 16
optimizer = 'Adam'  # ['SGD','Adam']
learning_rate = 0.01
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'  # ['multi', 'cos']
steps = [25, 38]
gamma = 0.1
warmup = 'linear'
warmup_iters = 64

# NETWORK
use_aux = False
griding_num = 32
backbone = '34'

# LOSS
sim_loss_w = 0
shp_loss_w = 0

# EXP
note = '-34-4lane-base-sim0-shp0'

log_path = '.\\checkpoints'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = '.\\checkpoints\\20220220_124726_lr_1e-02_b_16-34-4lane-base-sim0-shp0\\ep200.pth'
test_work_dir = None

num_lanes = 4
