import argparse
import os
import time
import sys
sys.path.append("./models/")

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from mobilenetv2 import MobileNetV2_cifar100, MobileNetV2_cifar10
from efficientnet import EfficientNetB0_cifar100, EfficientNetB0_cifar10
from shufflenet import ShuffleNetG2_cifar100, ShuffleNetG2_cifar10
from convnet import ConvNet, get_default_convnet_setting
from imagenet_ipc import ImageFolderIPC
from datetime import datetime

parser = argparse.ArgumentParser(description="UFC: Validate with Static Labeling")
parser.add_argument("--dataset", default="cifar100", type=str, choices=["cifar10", "cifar100"], help="Dataset")
parser.add_argument("--M", default=4, type=int, help="Number of architectures involved in UFC generation")
parser.add_argument("--networks", default='resnet18', type=str, help="Model architecture: resnet18, resnet34, resnet50, resnet101, mobilenetV2, efficientnet, shufflenet")
parser.add_argument("--epochs", default=200, type=int, help="Training epochs")
parser.add_argument("--batch-size", default=128, type=int, help="Batch size")
parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
parser.add_argument("--temperature", default=30, type=float, help="Temperature")
parser.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay")
parser.add_argument("--syn-data-path", default="./syn", type=str, help="Path to synthetic data")
parser.add_argument("--output-dir", default="./save", type=str, help="Directory to save results")
parser.add_argument("--resume", "-r", action="store_true", help="Resume from checkpoint")
parser.add_argument("--check-ckpt", default=None, type=str, help="Checkpoint to evaluate")
parser.add_argument("--ipc", default=5, type=int, help="IPC setting")
parser.add_argument('--wandb-project', type=str, default='UFC-validation', help='WandB project name')
parser.add_argument('--wandb-api-key', type=str, default=None, help='WandB API key')
parser.add_argument('--wandb-name', type=str, default="cifar100-ipc10", help='WandB run name')
 
args = parser.parse_args()

#init wandb 
wandb.login(key=args.wandb_api_key)
wandb.init(project=args.wandb_project, name=f"{args.wandb_name}_{datetime.now().strftime('%m/%d, %H:%M:%S')}")

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.check_ckpt:
    checkpoint = torch.load(args.check_ckpt)
    print(f"==> Loaded checkpoint: {args.check_ckpt}, Acc: {checkpoint['acc']}, Epoch: {checkpoint['epoch']}")
    exit()

os.makedirs(args.output_dir, exist_ok=True)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("==> Preparing data..")
dataset_config = {
    "cifar10": {
        "num_classes": 10,
        "transform_test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "transform_train": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    },
    "cifar100": {
        "num_classes": 100,
        "transform_test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]),
        "transform_train": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )        
    }
}

transform_test = dataset_config[args.dataset]["transform_test"]
transform_train = dataset_config[args.dataset]["transform_train"]
num_classes = dataset_config[args.dataset]["num_classes"]

args.ipc_init = int(args.ipc/(args.M/num_classes + 1))
args.ipc_total = args.ipc_init * (args.M + 1)

def check_files_per_folder(path, ipc_total):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")

    folder_counts = []

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):  
            num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            folder_counts.append(num_files)

            assert num_files == ipc_total, f"Error: Folder '{folder}' contains {num_files} files, expected {ipc_total}."

    if folder_counts:
        avg_files = sum(folder_counts) / len(folder_counts)
        print(f"✅ Average number of files per folder: {avg_files:.2f} (Expected: {ipc_total})")
    else:
        print("⚠ No subdirectories found.")

check_files_per_folder(args.syn_data_path, args.ipc_total)

trainset = ImageFolderIPC(root=args.syn_data_path, transform=transform_test, ipc=args.ipc_total)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root="../data", train=False, download=True, transform=transform_test) \
    if args.dataset == "cifar10" else \
    torchvision.datasets.CIFAR100(root="../data", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

print(f"==> Building model: {args.networks}")
model_classes = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "mobilenetV2": MobileNetV2_cifar100 if args.dataset == "cifar100" else MobileNetV2_cifar10,
    "efficientnet": EfficientNetB0_cifar100 if args.dataset == "cifar100" else EfficientNetB0_cifar10,
    "shufflenet": ShuffleNetG2_cifar100 if args.dataset == "cifar100" else ShuffleNetG2_cifar10,
}

if "resnet" in args.networks:
    model_student = model_classes[args.networks](num_classes=num_classes).to(device)
    model_student.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model_student.maxpool = nn.Identity()
else:
    model_student = model_classes[args.networks]().to(device)

if device == "cuda":
    model_student = torch.nn.DataParallel(model_student)
    cudnn.benchmark = True

# prepare archs for UFC
dataset_models = {
    'cifar10': {
        'num_classes': 10,
        'model_types': [torchvision.models.resnet18, MobileNetV2_cifar10, EfficientNetB0_cifar10, ShuffleNetG2_cifar10],
        'model_paths': [
            "pretrained/cifar10/resnet18_E200/ckpt.pth",
            "pretrained/cifar10/mobilenetV2_E200/ckpt.pth",
            "pretrained/cifar10/efficientnet_E200/ckpt.pth",
            "pretrained/cifar10/shufflenet_E200/ckpt.pth"
        ]
    },
    'cifar100': {
        'num_classes': 100,
        'model_types': [torchvision.models.resnet18, MobileNetV2_cifar100, EfficientNetB0_cifar100, ShuffleNetG2_cifar100],
        'model_paths': [
            "pretrained/cifar100/resnet18_E200/ckpt.pth",
            "pretrained/cifar100/mobilenetV2_E200/ckpt.pth",
            "pretrained/cifar100/efficientnet_E200/ckpt.pth",
            "pretrained/cifar100/shufflenet_E200/ckpt.pth"
        ]
    }
}

assert args.dataset in dataset_models, f"Unknown dataset: {args.dataset}"
dataset_config = dataset_models[args.dataset]
num_classes = dataset_config['num_classes']
model_types = dataset_config['model_types']
model_paths = dataset_config['model_paths']

model_lists = []
for model_type, model_path in zip(model_types, model_paths):
    if model_type == torchvision.models.resnet18:
        model_teacher = model_type(num_classes=num_classes)
        model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model_teacher.maxpool = nn.Identity() 
    else:
        model_teacher = model_type()

    model_teacher = nn.DataParallel(model_teacher).cuda()
    checkpoint = torch.load(model_path)
    model_teacher.load_state_dict(checkpoint["state_dict"])
    model_teacher.eval()

    model_lists.append(model_teacher)


# static labeling 
all_soft_labels = []
all_indices = []
start_index = 0
with torch.no_grad():
    for inputs, _, indices in trainloader:
        inputs = inputs.to(device)
        soft_label_avg = []
        for ii in range(len(model_types)):
            model_teacher = model_lists[ii]
            soft_label = model_teacher(inputs).detach()
            soft_label_avg.append(soft_label)
        soft_label = (soft_label_avg[0] + soft_label_avg[1] + soft_label_avg[2] + soft_label_avg[3]) / 4
        all_soft_labels.append(soft_label.clone().detach())
        batch_indices = indices
        all_indices.append(batch_indices)
        start_index += inputs.size(0)
all_soft_labels = torch.cat(all_soft_labels, dim=0)
all_indices = torch.cat(all_indices, dim=0)

# redefine train_loader
trainset = ImageFolderIPC(root=args.syn_data_path, transform=transform_train, ipc=args.ipc_total)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    model_student.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model_student.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
loss_function_kl = nn.KLDivLoss(reduction="batchmean")


def mixup_data(x, y, alpha=0.8):
    """
    Returns mixed inputs, mixed targets, and mixing coefficients.
    For normal learning
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index

# Train
def train(epoch,wandb_metrics):
    model_student.train()
    train_loss = 0
    correct = 0
    total = 0
    df1_sum = 0
    df2_sum = 0
    df3_sum = 0
    for batch_idx, (inputs, targets, indices) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        indices = indices.to(device)

        inputs, target_a, target_b, lam, mix_index = mixup_data(inputs, targets)

        soft_label = all_soft_labels[indices].to(device)
        soft_label = lam * soft_label + (1-lam)*soft_label[mix_index]

        optimizer.zero_grad()
        outputs = model_student(inputs)
        outputs_ = F.log_softmax(outputs / args.temperature, dim=1)

        soft_label_ = F.softmax(soft_label / args.temperature, dim=1)

        # crucial to make synthetic data and labels more aligned
        loss = loss_function_kl(outputs_, soft_label_)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f"Epoch: [{epoch}], Acc@1 {100.*correct/total:.3f}, Loss {train_loss/(batch_idx+1):.4f}")
    metrics = {
        "train/loss": float(f"{train_loss/(batch_idx+1):.4f}"),
        "train/Top1": float(f"{100.*correct/total:.3f}"),
        "train/epoch": epoch,
        "train/df1":float(f"{df1_sum:.4f}"),
        "train/df2":float(f"{df2_sum:.4f}"),
        "train/df3":float(f"{df3_sum:.4f}"),}
    wandb_metrics.update(metrics)
    wandb.log(wandb_metrics)


# Test
def test(epoch,wandb_metrics):
    global best_acc
    model_student.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_student(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"Test: Acc@1 {100.*correct/total:.3f}, Loss {test_loss/(batch_idx+1):.4f}")

    acc = 100.0 * correct / total
    if acc > best_acc:
        best_acc = acc

    metrics = {
        'val/loss': float(f"{test_loss/(batch_idx+1):.4f}"),
        'val/top1': float(f"{100.*correct/total:.3f}"),
        'val/epoch': epoch,
        'val/best_acc':best_acc,
    }
    wandb_metrics.update(metrics)
    wandb.log(wandb_metrics)
    print(f"Best: Acc@1 {best_acc:.3f}")



    # Save checkpoint.
    # save last checkpoint
    if True:
        state = {
            "state_dict": model_student.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')

        path = os.path.join(args.output_dir, "./ckpt.pth")
        torch.save(state, path)
        # best_acc = acc


start_time = time.time()
for epoch in range(start_epoch, start_epoch + args.epochs):
    global wandb_metrics
    wandb_metrics = {}

    train(epoch, wandb_metrics)
    # fast test
    if epoch % 10 == 0 or epoch == args.epochs - 1:
        test(epoch, wandb_metrics)
    scheduler.step()
end_time = time.time()
wandb.finish()
print(f"total time: {end_time - start_time} s")

