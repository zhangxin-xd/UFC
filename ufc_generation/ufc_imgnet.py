import os
import random
import collections
import numpy as np
import wandb
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data.distributed
import torchvision.models as models
from torchvision import transforms

from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.mobilenet import MobileNet_V2_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights

from PIL import Image
from utils import BNFeatureHook
from utils import clip_image, denormalize_image, save_images
from utils import lr_cosine_policy




def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    return preprocess(image)

def get_images(args, model_lists, ipc_id):
    # global wandb_metrics
    print("get_images call")
    save_every = 100
    batch_size = args.batch_size
    best_cost = 1e4

    loss_packed_features = [
        [BNFeatureHook(module) for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        for model in model_lists
    ]

    targets_all = torch.LongTensor(np.arange(1000))

    init_path = args.init_path
    for kk in range(0, 1000, batch_size):
        targets = targets_all[kk:min(kk + batch_size, 1000)].to('cuda')

        model_index = ipc_id // args.init - 1 - 1
        model_teacher = model_lists[model_index]
        loss_r_feature_layers = loss_packed_features[model_index]

        inputs_list = []
        sorted_folders = sorted(os.listdir(init_path))
        for folder_index in range(kk, min(kk + batch_size, 1000)):
            folder_path = os.path.join(init_path, sorted_folders[folder_index])
            image_path = os.path.join(folder_path, sorted(os.listdir(folder_path))[ipc_id%(args.ipc_init)])  
            image = load_image(image_path)  
            inputs_list.append(image)

        inputs_ori = torch.stack(inputs_list).to('cuda')  
        inputs_ori.requires_grad_(False)
        inputs_ori = inputs_ori.to('cuda')

        uni_perb = torch.zeros((1, 3, 224, 224), requires_grad=True, device="cuda", dtype=torch.float)
        uni_perb = uni_perb.to('cuda')


        iterations_per_layer = args.iteration if ipc_id >= args.ipc_init else 0
        inputs = input_original if iterations_per_layer == 0 else input_original + uni_perb

        lim_0, lim_1 = args.jitter , args.jitter

        optimizer = optim.Adam([uni_perb], lr=args.lr, betas=[0.5, 0.9], eps = 1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer) # 0 - do not use warmup
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)

            aug_func = transforms.Compose([
                transforms.RandomResizedCrop(224), # Crop Coord
                transforms.RandomHorizontalFlip(), # Flip Status
            ])

            inputs = inputs_ori + uni_perb
            inputs_jit = aug_func(inputs)
            # apply random jitter offsets
            off1 = random.randint(0, lim_0)
            off2 = random.randint(0, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

            # forward pass
            optimizer.zero_grad()
            outputs = model_teacher(inputs_jit)

            # R_cross classification loss
            loss_ce = criterion(outputs, targets)

            # R_feature loss
            rescale = [args.first_bn_multiplier] + [1.]* (len(loss_r_feature_layers) - 1)

            loss_r_bn_feature = [
                mod.r_feature.to(loss_ce.device) * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)
            ]
            loss_r_bn_feature = torch.stack(loss_r_bn_feature).sum()

            # combining losses
            loss_aux = args.r_bn * loss_r_bn_feature
            loss = loss_ce + loss_aux

            metrics = {
                'syn/loss_ce': loss_ce.item(),
                'syn/loss_aux': loss_aux.item(),
                'syn/loss_total': loss.item(),
                # 'syn/aux_weight': aux_weight,
                # 'image/acc_image': acc_image,
                # 'image/loss_image': loss_image,
                'syn/ipc_id': ipc_id,
            }

            # wandb_metrics.update(metrics)
            # wandb.log(wandb_metrics)

            if iteration % save_every==0:
                print("------------iteration {}----------".format(iteration))
                print("total loss", loss.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("main criterion", criterion(outputs, targets).item())
                # comment below line can speed up the training (no validation process)

            # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            inputs.data = clip_image(inputs.data, args.dataset)

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()

        if args.store_best_images:
            best_inputs = inputs.data.clone() # using multicrop, save the last one
            best_inputs = denormalize_image(best_inputs, args.dataset)
            save_images(args, best_inputs, targets, ipc_id)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
    torch.cuda.empty_cache()

def save_images(args, images, targets, ipc_id):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store = dir_path +'/class{:03d}_id{:03d}.jpg'.format(class_id,ipc_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)

def validate(input, target, model):
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())

def generation(args, ipc_id):

    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)
   
    model_types = {
        'resnet18': ResNet18_Weights.IMAGENET1K_V1,
        'mobilenet_v2': MobileNet_V2_Weights.IMAGENET1K_V1,
        'efficientnet_b0': EfficientNet_B0_Weights.IMAGENET1K_V1
    }

    model_lists = []

    for model_name, weight in model_types.items():
        model_teacher = models.__dict__[model_name](weights=weight)

        model_teacher = nn.DataParallel(model_teacher).cuda()
        model_teacher.eval()

        for p in model_teacher.parameters():
            p.requires_grad = False

        model_lists.append(model_teacher)

    get_images(args, model_lists, ipc_id)

def get_args():
    parser = argparse.ArgumentParser(description="UFC: Generate Inter-class Feature Compensator")
    
    # General settings
    parser.add_argument("--dataset", default="imagenet", type=str, choices=["cifar10", "cifar100", "imagenet"],
                        help="Dataset selection: cifar10, cifar100, or imagenet")
    parser.add_argument("--M", default=3, type=int, help="Number of architectures involved in UFC generation")
    parser.add_argument("--init_path", type=str, default="",
                        help="Path to the initial synthetic data")
    parser.add_argument("--ipc", type=int, default=10,
                        help="IPC (images per class) setting")

    # Data saving parameters
    parser.add_argument("--exp-name", type=str, default="generated_results",
                        help="Experiment name (subfolder under --syn-data-path)")
    parser.add_argument("--wandb-name", type=str, default="imagenet-ipc10")
    parser.add_argument("--syn-data-path", type=str, default="./syn",
                        help="Root directory for storing synthetic data")
    parser.add_argument("--store-best-images", action="store_true",
                        help="Flag to store the best-generated images")

    # Optimization parameters
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of images optimized simultaneously")
    parser.add_argument("--iteration", type=int, default=1000,
                        help="Number of iterations for optimization")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for optimization")
    parser.add_argument("--jitter", type=int, default=32,
                        help="Random shift applied to synthetic data for augmentation")
    parser.add_argument("--r-bn", type=float, default=0.05,
                        help="Coefficient for batch normalization (BN) feature distribution regularization")
    parser.add_argument("--first-bn-multiplier", type=float, default=10.0,
                        help="Additional multiplier for the first BN layer in R_bn")

    
    # Parse arguments
    args = parser.parse_args()

    # Update syn_data_path to include experiment name
    args.syn_data_path = os.path.join(args.syn_data_path, args.wandb_name, args.exp_name)

    return args
 

if __name__ == '__main__':

    args = get_args()

    if not wandb.api.api_key:
        wandb.login(key='')
    wandb.init(project='UFC-generation', name=args.wandb_name)
    global wandb_metrics
    wandb_metrics = {}
    args.ipc_start = 0
    if args.dataset =='cifar10':
        args.num_class = 10
    elif args.dataset =='cifar100':
        args.num_class = 100
    elif args.dataset =='imagenet':
        args.num_class = 1000

    # averaging UFC for fair comparison
    args.ipc_init = int(args.ipc/(args.M/args.num_class + 1))
    args.ipc_end = args.ipc_init * (args.M + 1)
    print('ipc_end = ', args.ipc_end)

    for ipc_id in range(args.ipc_start, args.ipc_end):
        print("ipc = ", ipc_id)
        wandb.log({'ipc_id': ipc_id})
        generation(args, ipc_id)

    wandb.finish()