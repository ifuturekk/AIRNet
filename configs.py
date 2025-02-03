import torch
from torchvision import transforms
from dataloader.imaging import gray, stft, cwt, rp, gasf, gadf, mtf


class Config(object):
    def __init__(self):
        self.data_folder = r'./data'
        self.data_path = r'./dataloader/data'
        self.exp_log_dir = 'experiment_logs'
        self.file_names = ['comp1.csv', 'comp2.csv', 'comp3.csv', 'comp4.csv', 'comp5.csv', 'comp6.csv']
        self.faults = ['排气压力', '吸气压力']
        self.higher = [True, False]
        self.thresholds = [980, 220]
        self.time_len = 360

        self.domains = [1, 2, 3, 4, 5, 6]
        self.tasks = [([source for source in self.domains if source != target], target) for target in self.domains]
        self.train_transform = None  # transforms.Compose([transforms.ToTensor()])
        self.valid_transform = None  # transforms.Compose([transforms.ToTensor()])

        self.imaging = cwt  # None, gray, stft, cwt
        self.cutmix_ab = (1.0, 1.0)

        self.in_channels = 8
        self.out_channels = 32
        self.dropout = 0.1
        self.feature_len = 32
        self.feature_matrix = 1  # Global Average Pooling (GAP)

        self.attention = 'se'  # None, 'se', 'SimAM'
        self.reduction = 16

        self.train_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.eval_device = torch.device('cpu')
        self.batch_size = 1024
        self.pre_train_epoch = 20
        self.train_epoch = 100
        self.fine_tune_epoch = 100
        self.aug_num = 16
        self.repeat_num = 10

        self.Conv1D = Conv1d()
        self.VGG = VGG()
        self.ResNet = ResNet()


class VGG(object):
    def __init__(self):
        # raw: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.cfg = [4, 'M', 8, 'M', 16, 16, 'M', 32, 32, 'M', 32, 32, 'M']


class ResNet(object):
    def __init__(self):
        # raw: [64, 64, 128, 256, 512]
        # self.cfg = [4, 4, 8, 16, 32]
        self.cfg = [16, 16, 32]


class Conv1d(object):
    def __init__(self):
        self.kernel_size = 8
        self.stride = 2
        self.hidden_channels = [16, 32, 64]


