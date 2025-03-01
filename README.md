# 🌟 Breaking Class Barriers: Efficient Dataset Distillation via Inter-Class Feature Compensator
## 🔥 ICLR 2025 Poster 🔥

>[Breaking Class Barriers: Efficient Dataset Distillation via Inter-Class Feature Compensator](https://arxiv.org/abs/2408.06927).<br>
> [Xin Zhang](https://zhangxin-xd.github.io/), [Jiawei Du](https://scholar.google.com/citations?user=WrJKEzEAAAAJ&hl=zh-CN), [Ping Liu](https://pinglmlcv.github.io/pingliu264/), [Joey Tianyi Zhou](https://joeyzhouty.github.io/) <br>
> Agency for Science, Technology, and Research (ASTAR), Singapore <br>
> University of Nevada, Reno
## 📖 Introduction
<p align="justify">
Dataset distillation has emerged as a technique aiming to condense informative features from large, natural datasets into a compact and synthetic form. While recent advancements have refined this technique, its performance is bottlenecked by the prevailing class-specific synthesis paradigm. Under this paradigm, synthetic data is optimized exclusively for a pre-assigned one-hot label, creating an implicit class barrier in feature condensation. This leads to inefficient utilization of the distillation budget and oversight of inter-class feature distributions, which ultimately limits the effectiveness and efficiency, as demonstrated in our analysis.
To overcome these constraints, this paper presents the Inter-class Feature Compensator (INFER), an innovative distillation approach that transcends the class-specific data-label framework widely utilized in current dataset distillation methods. Specifically, INFER leverages a Universal Feature Compensator (UFC) to enhance feature integration across classes, enabling the generation of multiple additional synthetic instances from a single UFC input. This significantly improves the efficiency of the distillation budget.
Moreover, INFER enriches inter-class interactions during the distillation, thereby enhancing the effectiveness and generalizability of the distilled data. By allowing for the linear interpolation of labels similar to those in the original dataset, INFER meticulously optimizes the synthetic data and dramatically reduces the size of soft labels in the synthetic dataset to almost zero, establishing a new benchmark for efficiency and effectiveness in dataset distillation. In practice, INFER demonstrates state-of-the-art performance across benchmark datasets. For instance, in the $\texttt{ipc} = 50$ setting on ImageNet-1k with the same compression level, it outperforms SRe2L by 34.5% using ResNet18.</p>

---

## ⚙️ Installation

To get started, follow these instructions to set up the environment and install dependencies.

1. **Clone this repository**:
    ```bash
    git clone https://github.com/zhangxin-xd/UFC.git
    cd UFC
    ```

2. **Install required packages**:
   You don’t need to create a new environment; simply ensure that you have compatible versions of CUDA and PyTorch installed.
---

## 🚀 Usage

Here’s how to use this code for UFC generation and validation:
- **Preparation**
For ImageNet-1K, we use the pre-trained weights available in `torchvision`.  For CIFAR and Tiny-ImageNet, we provide the trained weights at this [link](https://drive.google.com/drive/folders/1dH96COYa4kCquQ4c6wEnt7QobGMl6M3N?usp=sharing).  Alternatively, you can train the models yourself by following the instructions in [Diversity-Driven-Synthesis](https://github.com/AngusDujw/Diversity-Driven-Synthesis).

- **Distillation**:
    Before performing distillation, please first prepare the images by randomly sampling from the original dataset and saving them as tensors. We provide the tensor-formatted initialization images at this [link](https://drive.google.com/drive/folders/1ueAnTXOUGiQ_E9iIssNYmEBX4vlVQEDZ?usp=sharing) .

    Cifar:
    ```bash
    python distillation/distillation_cifar.py 
        --iteration 1000 --r-bn 0.01 --batch-size 100 --lr 0.25 
        --exp-name distillation-c100-ipc50 
        --store-best-images 
        --syn-data-path ./syn_data/ 
        --init_path ./distillation/init_images/cifar100 
        --steps 12 --rho 15e-3 --ipc-start 0 --ipc-end 50 --r-var 11 
        --dataset cifar100 
    ```
    Tiny-ImageNet:
    ```bash
    python distillation/distillation_tiny.py 
         --iteration 2000 --r-bn 0.01 --batch-size 200 --lr 0.1 
         --exp-name distillation-tiny-ipc50 
         --store-best-images 
         --syn-data-path ./syn_data/ 
         --init-path ./distillation/init_images/tiny 
         --steps 12 --rho 15e-3 --ipc-start 0 --ipc-end 50 --r-var 11 
         --dataset tiny 
    ```
    ImageNet-1K:
    ```bash
    python distillation/distillation_imgnet.py 
        --exp-name distillation-imgnet-ipc50  
        --syn-data-path ./syn_data/ 
        --init-path ./distillation/init_images/imgnet/ 
        --arch-name resnet18 
        --batch-size 100 --lr 0.25 --iteration 2000 --r-bn 0.01 
        --r-var 2 --steps 15 --rho 15e-3 
        --store-best-images 
        --ipc-start 0 --ipc-end 50 
    ```
- **Evaluation**:
  
    Cifar:
    ```bash
    python validation/validation_cifar.py 
          --epochs 400 --batch-size 128 --ipc 10 
          --syn-data-path ./syn_data/distillation-c100-ipc50 
          --output-dir ./syn_data/validation-c100-ipc50 
          --networks resnet18 --dataset cifar100 
    ```
    Tiny-ImageNet:
    ```bash
    python validation/validation_tiny.py 
            --epochs 200 --batch-size 64 --ipc 50 
            --lr 0.2 --momentum 0.9 --weight-decay 1e-4 
            --lr-scheduler cosineannealinglr 
            --lr-warmup-epochs 5 
            --lr-warmup-method linear 
            --lr-warmup-decay 0.01
            --data-path ./data/tiny/
            --syn-data-path ./syn_data/distillation-tiny-ipc50/ 
            --model resnet18   
    ```
    ImageNet-1K:
    ```bash
    python validation/validation_imgnet.py 
        --epochs 300 --batch-size 128 --ipc 50 
        --mix-type cutmix 
        --cos -T 20 -j 4 
        --train-dir ./syn_data/distillation-imgnet-ipc50 
        --output-dir ./syn_data/validation-imgnet-ipc50 
        --val-dir ./data/Imagenet-1k/val 
        --teacher-model resnet18 
        --model resnet18 
    ```
we also provide the `.sh` script in the `scripts` directory.

---

## 📊 Results

Our experiments demonstrate the effectiveness of the proposed approach across various benchmarks. 

![Results](./imgs/results.png)

For detailed experimental results and further analysis, please refer to the full paper.

---

## 📑 Citation

If you find this code useful in your research, please consider citing our work:

```bibtex
@inproceedings{dwa2024neurips,
    title={Diversity-Driven Synthesis: Enhancing Dataset Distillation through Directed Weight Adjustment},
    author={Du, Jiawei and Zhang, Xin and Hu, Juncheng and Huang, Wenxin and Zhou, Joey Tianyi},
    booktitle={Adv. Neural Inf. Process. Syst. (NeurIPS)},
    year={2024}
}
```
---
## 🎉 Reference
Our code has referred to previous work:

[Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective](https://github.com/VILA-Lab/SRe2L)

