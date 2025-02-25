from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, CamVid_Dataset, COCOSegmentation, Test, ADE20KSegmentation, COCOSegmentation, SPSegmentation, NewDefineSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from torchvision import transforms


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./data/Cityscapes/',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes', 'camvid', 'mscoco', 'ade', 'sps'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    parser.add_argument("--pretrained", type=str, default='./',
                        help="path to pretrained weight")
    parser.add_argument("--model", type=str, default='AFASeg_S',
                        help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int,
                        default=16, choices=[8, 16])
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='cosinewm', choices=['poly', 'step', 'cosinewm'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512,
                        help="crop size for voc pascal")

    parser.add_argument("--city_crop_size", type=int,
                        default=[512, 1024], help="crop size for cityscapes & camvid datatset")

    parser.add_argument("--camvid_crop_size", type=int,
                        default=[720, 960], help="crop size for cityscapes & camvid datatset")

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training",
                        action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0,1',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(
                size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            et.ExtRandomCrop(
                size=(opts.city_crop_size[0], opts.city_crop_size[1])),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize((opts.city_crop_size[0], opts.city_crop_size[1])),
                et.ExtCenterCrop(
                    (opts.city_crop_size[0], opts.city_crop_size[1])),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)

    if opts.dataset == 'camvid':
        train_transform = et.ExtCompose([
            et.ExtRandomCrop(
                size=(opts.camvid_crop_size[0], opts.camvid_crop_size[1])),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(
                    (opts.camvid_crop_size[0], opts.camvid_crop_size[1])),
                et.ExtCenterCrop(
                    (opts.camvid_crop_size[0], opts.camvid_crop_size[1])),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

        train_dst = CamVid_Dataset(img_pth=opts.data_root + '/train/', mask_pth=opts.data_root + '/train_labels/',
                                   transform=train_transform)
        val_dst = CamVid_Dataset(img_pth=opts.data_root + '/val/', mask_pth=opts.data_root + '/val_labels/',
                                 transform=val_transform)

    if opts.dataset == 'mscoco':
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        train_dst = COCOSegmentation(
            root=opts.data_root, split='train', transform=input_transform)
        val_dst = COCOSegmentation(
            root=opts.data_root, split='val', transform=input_transform)

    if opts.dataset == 'ade':
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        # Create Dataset
        train_dst = ADE20KSegmentation(root=opts.data_root,
                                       split='train',
                                       transform=input_transform)
        val_dst = ADE20KSegmentation(root=opts.data_root,
                                     split='val',
                                     transform=input_transform)
    if opts.dataset == 'sps':
        train_transform = et.ExtCompose([
            et.ExtResize((opts.crop_size + 32, opts.crop_size + 32)),
            # et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(
                size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize((opts.crop_size, opts.crop_size)),
                # et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = SPSegmentation(
            root=opts.data_root, image_set='train', transform=train_transform)
        val_dst = SPSegmentation(
            root=opts.data_root, image_set='val', transform=val_transform)

    return train_dst, val_dst


def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            if opts.dataset.lower() == 'camvid':
                labels = labels.argmax(1)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1,
                                                            2, 0).astype(np.uint8)
                    if opts.dataset.lower() == 'camvid':
                        label = labels[i].squeeze()
                        target = loader.dataset.decode_segmap(target)
                        pred = loader.dataset.decode_segmap(pred)
                    else:
                        target = loader.dataset.decode_target(
                            target).astype(np.uint8)
                        pred = loader.dataset.decode_target(
                            pred).astype(np.uint8)

                    Image.fromarray(image).save(
                        'results/%d_image.png' % img_id)
                    Image.fromarray(target).save(
                        'results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' %
                                img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'camvid':
        opts.num_classes = 11
    elif opts.dataset.lower() == 'mscoco':
        opts.num_classes = 171
    elif opts.dataset.lower() == 'ade':
        opts.num_classes = 150
    elif opts.dataset.lower() == 'sps':
        opts.num_classes = 2

    from datetime import datetime
    path = os.path.join('./results/',
                        opts.dataset + '_' +
                        opts.model + '_' +
                        str(datetime.today().month) + '.' +
                        str(datetime.today().day) + '.' +
                        str(datetime.today().hour) + '.' +
                        str(datetime.today().minute)
                        )
    print(path)

    if not os.path.isdir(path):
        os.makedirs(path)
    log_file = open(path + '/log.txt', "w")

    model_file = open(path + '/model_summary.txt', "w")
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=8,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=8)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes, output_stride=opts.output_stride)
    ######################
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)

    from torchinfo import summary
    if opts.dataset == 'voc' or opts.dataset == 'mscoco':
        summary(model, (1, 3, opts.crop_size, opts.crop_size))
        model_file.write(
            str(summary(model, (1, 3, opts.crop_size, opts.crop_size))) + '\n')
    elif opts.dataset == 'camvid':
        summary(
            model, (1, 3, opts.camvid_crop_size[0], opts.camvid_crop_size[1]))
        model_file.write(str(summary(
            model, (1, 3, opts.camvid_crop_size[0], opts.camvid_crop_size[1]))) + '\n')
    else:
        summary(
            model, (1, 3, opts.city_crop_size[0], opts.city_crop_size[1]), device=device)
        model_file.write(str(summary(
            model, (1, 3, opts.city_crop_size[0], opts.city_crop_size[1]), device=device)) + '\n')
    model_file.write('\ncrop_val : {}'.format(opts.crop_val))
    model_file.close()
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    if opts.dataset == 'ade':
        optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opts.step_size, gamma=0.1)
    elif opts.lr_policy == 'cosinewm':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=opts.total_itrs, cycle_mult=0.5,
                                                  max_lr=opts.lr, min_lr=opts.lr*0.001, warmup_steps=6550, gamma=0.8, last_epoch=-1)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        if opts.dataset == 'ade':
            criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            # "model_state": model.module.state_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')

    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint

        model = model.to(device)
        model = nn.DataParallel(model)
    else:
        print("[!] Retrain")
        print(device)
        model = model.to(device)
        model = nn.DataParallel(model)

    # ==========   Train Loop   ==========#

    if opts.test_only:
        checkpoint = torch.load(
            opts.pretrained, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        print("CheckPoint is loaded,,,,")
        replace_batchnorm(model)
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            if opts.dataset.lower() == 'camvid':
                labels = labels.argmax(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f, Learning Rate: %lf" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss, optimizer.param_groups[0]['lr']))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt(path + '/latest_%s_%s.pth' %
                          (opts.model, opts.dataset))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
                print(metrics.to_str(val_score))
                log_file.write('Mean IoU : {}\n'.format(val_score['Mean IoU']))
                log_file.close()
                log_file = open(path + '/log.txt', "a")
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(path + '/best_%s_%s.pth' %
                              (opts.model, opts.dataset))
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                print("Best Mean IoU: ", best_score)
                return


if __name__ == '__main__':
    main()
