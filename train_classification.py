from __future__ import print_function
from __future__ import division

import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from datasets import load_dataset
from torchvision import transforms
import time
import copy

import os

import wandb
from detectron2.config import get_cfg

from model import add_kd_config
from model.engine.trainer import KdTrainer

# hacky way to register
from model.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from model.modeling.proposal_generator.rpn import PseudoLabRPN
from model.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import model.data.datasets.builtin

from model.modeling.meta_arch.ts_ensemble import EnsembleTSModel

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PreTrainingBackboneForImageClassification(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x['p5'])
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def training_step(phase, model, dataloaders, optimizer, is_inception, criterion, best_acc, best_model_wts, output_dir, epoch, val_acc_history):
    if phase == 'train':
        model.train()  # Set model to training mode
    else:
        model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for sample in dataloaders[phase]:
        inputs = sample["pixel_values"].to(device)
        labels = sample["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            # Get model outputs and calculate loss
            # Special case for inception because in training it has an auxiliary output. In train
            #   mode we calculate the loss by summing the final output and the auxiliary output
            #   but in testing we only consider the final output.
            if is_inception and phase == 'train':
                # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    wandb.log({f'{phase}_loss': epoch_loss, f'{phase}_acc': epoch_acc})

    # deep copy the model
    if phase == 'validation' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        os.makedirs(os.path.join("runs", output_dir), exist_ok=True)
        torch.save(best_model_wts, os.path.join(f"runs/{output_dir}", f'resnet50_epoch_{epoch}.pth'))
    if phase == 'validation':
        val_acc_history.append(epoch_acc)
    if phase == 'test':
        best_acc = epoch_acc
    return best_acc, best_model_wts


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, output_dir="default"):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            best_acc, best_model_wts = training_step(phase, model, dataloaders, optimizer, is_inception, criterion, best_acc, best_model_wts, output_dir, epoch, val_acc_history)

        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print()
    print('Test Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, os.path.join("runs", 'best_model.pth'))
    print('Testing')
    best_acc = training_step('test', model, dataloaders, optimizer, is_inception, criterion, best_acc, output_dir, epoch, val_acc_history)
    return model, val_acc_history


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def transforms_fn(example_batch, split):
    """Apply _train_transforms across a batch."""
    example_batch["pixel_values"] = [
        data_transforms[split](pil_img.convert("RGB")) for pil_img in example_batch['image']
    ]
    return example_batch


def main(cfg, dataset_name, shot, seed):
    dataset_path = f"HichTala/{dataset_name}_{shot}shot_{seed}"
    dataset = load_dataset(dataset_path)
    num_classes = len(dataset["train"].features["label"].names)

    Trainer = KdTrainer
    detection_model = Trainer.build_model(cfg)
    model = PreTrainingBackboneForImageClassification(detection_model.backbone)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Send the model to GPU
    model = model.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    print("Params to learn:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", name)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=0.0001, momentum=0.9)

    print("Initializing Datasets and Dataloaders...")

    dataset["train"].set_transform(partial(transforms_fn, split="train"))
    dataset["validation"].set_transform(partial(transforms_fn, split="validation"))
    dataset["test"].set_transform(partial(transforms_fn, split="test"))

    dataloaders = {
        split_name: torch.utils.data.DataLoader(dataset[split_name], batch_size=32, shuffle=True, num_workers=3,
                                                collate_fn=collate_fn)
        for split_name in ['train', 'validation', 'test']
    }

    criterion = nn.CrossEntropyLoss()

    train_model(model, dataloaders, criterion, optimizer, num_epochs=30,
                output_dir=f"{dataset_name}_{shot}shot_{seed}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset to use for training')
    parser.add_argument('--shot', required=True, help='dataset to use for training')
    parser.add_argument('--seed', required=True, help='dataset to use for training')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    input_size = 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    cfg = get_cfg()
    add_kd_config(cfg)
    cfg.merge_from_file(f"./configs/{args.dataset}/{args.shot}_shot.yaml")
    cfg.merge_from_list([])
    cfg.freeze()

    wandb.init(project=f"ICIP-2026-cdfsod_{args.dataset}", group=f"{args.shot}_shots")

    main(cfg, dataset_name=args.dataset, shot=args.shot, seed=args.seed)
