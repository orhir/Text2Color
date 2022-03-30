import argparse
import os
import numpy as np
from clip.clip import BICUBIC
from skimage import color, io
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision.transforms import Compose
from tqdm import tqdm
import CLIP_prefix_caption.predict
from models import ColorUNet
from text2color import Text2Color
from vgg_model import vgg19
from data.data_loader import MultiResolutionDataset
import matplotlib.pyplot as plt
from utils import tensor_lab2rgb
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
)
from transformers import GPT2Tokenizer
import torchvision.transforms as transforms
import clip
from deephist.hist_layers import SingleDimHistLayer, JointHistLayer
from deephist.metrics import EarthMoversDistanceLoss, MutualInformationLoss
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import torch.optim as optim


def firstRepeat(sentence):
    words = sentence.split(' ')
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            return i + 1
    return len(words)


def mkdirss(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def data_sampler(data_set, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(data_set, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(data_set)
    else:
        return data.SequentialSampler(data_set)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def Lab2RGB_out(img_lab):
    img_lab = img_lab.detach().cpu()
    img_l = img_lab[:, :1, :, :]
    img_ab = img_lab[:, 1:, :, :]
    img_l = img_l + 50
    pred_lab = torch.cat((img_l, img_ab), 1)[0, ...].numpy()
    out = (np.clip(color.lab2rgb(pred_lab.transpose(1, 2, 0)), 0, 1) * 255).astype("uint8")
    return out


def RGB2Lab(inputs):
    # input [0, 255] uint8
    # out l: [0, 100], ab: [-110, 110], float32
    return color.rgb2lab(inputs)


def Normalize(inputs):
    l = inputs[:, :, 0:1]
    ab = inputs[:, :, 1:3]
    l = l - 50
    lab = np.concatenate((l, ab), 2)

    return lab.astype('float32')


def numpy2tensor(inputs):
    out = torch.from_numpy(inputs.transpose(2, 0, 1))
    return out


def tensor2numpy(inputs):
    out = inputs[0, ...].detach().cpu().numpy().transpose(1, 2, 0)
    return out


def preprocessing(inputs):
    # input: rgb, [0, 255], uint8
    img_lab = Normalize(RGB2Lab(inputs))
    img = np.array(inputs, 'float32')  # [0, 255]
    img = numpy2tensor(img)
    img_lab = numpy2tensor(img_lab)
    return img.unsqueeze(0), img_lab.unsqueeze(0)


def uncenter_l(inputs):
    l = inputs[:, :1, :, :] + 50
    ab = inputs[:, 1:, :, :]
    return torch.cat((l, ab), 1)


def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_last_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


def train(
        args,
        loader,
        text2Color,
        colorUNet,
        vggnet,
        g_optim,
        device,
        scheduler
):
    # torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter()

    loader = sample_data(loader)

    pbar = range(args.iter // args.batch)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    loss_dict = {}

    if args.distributed:
        text2Color_module = text2Color.module
        colorUNet_module = colorUNet.module

    else:
        text2Color_module = text2Color
        colorUNet_module = colorUNet

    clip_model, clip_preprocess = clip.load("RN50x4", device=device, jit=False)

    prefix_length = 40
    caption_model = CLIP_prefix_caption.predict.ClipCaptionPrefix(prefix_length, clip_length=40, prefix_size=640,
                                                                  num_layers=8,
                                                                  mapping_type=CLIP_prefix_caption.predict.MappingType.Transformer)
    caption_model.load_state_dict(torch.load("CLIP_prefix_caption/transformer_weights.pt"))
    caption_model = caption_model.eval()
    caption_model = caption_model.to(device)

    # clip_size = 288 if args.auto_caption else 224
    clip_size = 288
    clip_transforms = torch.nn.Sequential(
        transforms.RandomResizedCrop(clip_size, scale=(1, 1)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        transforms.RandomHorizontalFlip(),
    )
    _clip_preprocess = Compose([
        transforms.Resize(clip_size, interpolation=BICUBIC),
        transforms.CenterCrop(clip_size),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    requires_grad(clip_model, False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    loss_list_total = []
    loss_list_rec = []
    loss_list_feat = []
    loss_list_clip = []
    loss_list_hist = []

    out_dir = "experiments/%s" % args.experiment_name
    mkdirss(out_dir)

    recon_val_avg = 0
    fea_val_avg = 0
    clip_val_avg = 0
    hist_val_avg = 0
    loss_avg = 0

    L1_loss = nn.L1Loss()
    L2_loss = nn.MSELoss()

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter / args.batch:
            print("Done!")

            break

        _img, img_caption, img_lab = next(loader)

        img = _img.to(device)  # GT [B, 3, 256, 256]
        img_lab = img_lab.to(device)  # GT

        img_l = img_lab[:, :1, :, :] / 50  # [-1, 1] target L
        img_ab = img_lab[:, 1:, :, :] / 110  # [-1, 1] target ab
        real_img_rgb = img / 255.
        if args.auto_caption:
            img_caption = []
        with torch.no_grad():
            for b_image in real_img_rgb[:, ...]:
                encoded_real_image = clip_model.encode_image(_clip_preprocess(b_image.unsqueeze(0)).to(device))
                if args.auto_caption:
                    prefix = encoded_real_image.to(device, dtype=torch.float32)
                    prefix = prefix / prefix.norm(2, dim=1)[:, None]
                    prefix_embed = caption_model.clip_project(prefix).reshape(1, prefix_length, -1)
                    caption = CLIP_prefix_caption.predict.generate_beam(caption_model, tokenizer, embed=prefix_embed)[0]
                    caption = caption.replace("black and white", "")
                    if len(caption) > 50:
                        caption = " ".join(caption.split()[:firstRepeat(caption)])[:50]
                    img_caption.append(caption)
            z_caption = clip_model.encode_text(clip.tokenize(img_caption).to(device))
            # original_clip_similarity = torch.mean(torch.cosine_similarity(encoded_real_image, z_caption))

        text2Color.train()
        colorUNet.train()

        requires_grad(text2Color, True)
        requires_grad(colorUNet, True)
        ref_color_vector = text2Color(z_caption)
        fake_swap_ab = colorUNet((img_l, ref_color_vector))  # [-1, 1]

        fake_swap_rgb = tensor_lab2rgb(torch.cat((img_l * 50 + 50, fake_swap_ab * 110), 1))  # [0, 1]
        ## recon l1 loss
        recon_loss = (F.smooth_l1_loss(fake_swap_ab, img_ab))

        ## feature loss
        fea_loss = torch.zeros(1).to(device)

        features_A = vggnet(real_img_rgb, layer_name='all')
        features_B = vggnet(fake_swap_rgb, layer_name='all')

        fea_loss1 = L2_loss(features_A[0], features_B[0]) / 32
        fea_loss2 = L2_loss(features_A[1], features_B[1]) / 16
        fea_loss3 = L2_loss(features_A[2], features_B[2]) / 8
        fea_loss4 = L2_loss(features_A[3], features_B[3]) / 4
        fea_loss5 = L2_loss(features_A[4], features_B[4])

        fea_loss = (fea_loss1 + fea_loss2 + fea_loss3 + fea_loss4 + fea_loss5) * 0.1

        ## Clip loss
        clip_loss = torch.zeros(1).to(device)
        if i > 40000 / args.batch:
            # for _ in range(args.n_aug):
            # rgb_fake_img = clip_transforms(fake_swap_rgb)
            encoded_image = clip_model.encode_image(_clip_preprocess(fake_swap_rgb))
            fake_clip_similarity = torch.mean(torch.cosine_similarity(encoded_image, z_caption))
            clip_loss = (1 - fake_clip_similarity) * args.lambda_clip
            # clip_loss = (1 / args.n_aug) * clip_loss * args.lambda_clip

        ## Histogram Loss
        hist_loss = torch.zeros(1).to(device)
        if i > 1700008 / args.batch:
            img_a = img_ab[:, 0, :, :]
            hist_a = SingleDimHistLayer(device)(img_a)
            fake_a = fake_swap_ab[:, 0, :, :]
            hist_fake_a = SingleDimHistLayer(device)(fake_a)
            img_b = img_ab[:, 1, :, :].to(device)
            hist_b = SingleDimHistLayer(device)(img_b)
            fake_b = fake_swap_ab[:, 1, :, :]
            hist_fake_b = SingleDimHistLayer(device)(fake_b)

            emd_hist_loss_a = EarthMoversDistanceLoss(device)(hist_a, hist_fake_a).mean()
            emd_hist_loss_b = EarthMoversDistanceLoss(device)(hist_b, hist_fake_b).mean()

            hist_loss = (emd_hist_loss_a + emd_hist_loss_b) * args.lambda_hist

        loss_dict["recon"] = recon_loss
        loss_dict["fea"] = fea_loss
        loss_dict["clip"] = clip_loss
        loss_dict["hist_loss"] = hist_loss

        loss = recon_loss + fea_loss + clip_loss + hist_loss

        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

        loss_reduced = reduce_loss_dict(loss_dict)

        recon_val = loss_reduced["recon"].mean().item()
        fea_val = loss_reduced["fea"].mean().item()
        clip_val = loss_reduced["clip"].mean().item()
        hist_loss_val = loss_dict["hist_loss"].mean().item()

        # recon_val_avg += recon_val
        # fea_val_avg += fea_val
        # clip_val_avg += clip_val
        # hist_val_avg += hist_loss_val
        # loss_avg += loss.item()

        writer.add_scalar('Loss/Reconstruction', recon_val, idx)
        writer.add_scalar('Loss/Feature', fea_val, idx)
        writer.add_scalar('Loss/Clip', clip_val, idx)
        writer.add_scalar('Loss/Hist', hist_loss_val, idx)
        writer.add_scalar('Loss/Total', loss.item(), idx)

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"recon:{recon_val:.6f}; fea:{fea_val:.6f}; clip:{clip_val:.6f};"
                    f"hist:{hist_loss_val:.6f}"
                )
            )
            if i % \
                    (args.print_freq // args.batch) == 0:
                # scheduler.step()

                new_lr = adjust_lr(g_optim, idx, 1e-4, 0.99, scheduler)
                print('lr: {0:.3e}'.format(new_lr))

                # div = args.print_freq if i > 0 else 1
                # loss_list_total.append(loss_avg / div)
                # loss_list_rec.append(recon_val_avg / div)
                # loss_list_feat.append(fea_val_avg / div)
                # loss_list_clip.append(clip_val_avg / div)
                # loss_list_hist.append(hist_val_avg / div)
                # loss_img_path = os.path.join(out_dir, 'LOSS.png')

                # x = np.arange(len(loss_list_total)) * args.batch
                # plt.clf()
                # plt.plot(x, loss_list_total, label='Total Loss')
                # plt.plot(x, loss_list_rec, label='Rec Loss')
                # plt.plot(x, loss_list_feat, label='Feature Loss')
                # plt.plot(x, loss_list_clip, label='Clip Loss')
                # plt.plot(x, loss_list_hist, label='Hist Loss')
                # plt.xlabel("Iters")
                # plt.ylabel("Loss")
                # plt.legend(loc='center right', bbox_to_anchor=(1, 0))
                # plt.savefig(loss_img_path, bbox_inches="tight")
                # out_img_path = os.path.join(out_dir, 'in{}_{}.png'.format(i, img_caption[0].replace(" ", "_")))
                # out_img = (fake_swap_rgb[0, ...].squeeze(0).cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(
                #     "uint8")
                # io.imsave(out_img_path, out_img)

                writer.add_images('Original_Image', real_img_rgb.cpu().detach(), idx)
                writer.add_images('Colored_Image', fake_swap_rgb.cpu().detach(), idx)

                # loss_avg = 0
                # recon_val_avg = 0
                # fea_val_avg = 0
                # clip_val_avg = 0
                # hist_val_avg = 0

            if i % (args.save_freq // args.batch) == 0:
                torch.save(
                    {
                        "text2Color": text2Color_module.state_dict(),
                        "colorUNet": colorUNet_module.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "args": args,
                    },
                    f"%s/{str(i).zfill(6)}.pt" % out_dir,
                )


if __name__ == "__main__":
    device = "cuda"

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--iter", type=int, default=2000000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--print_freq", type=int, default=5000)
    parser.add_argument("--save_freq", type=int, default=50000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--lambda_clip", type=float, default=0.4)
    parser.add_argument("--lambda_hist", type=float, default=0.0005)
    parser.add_argument("--n_aug", type=int, default=5)
    parser.add_argument("--load_only_colorUNet", action='store_true')
    parser.add_argument("--auto_caption", action='store_true')

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.start_iter = 0

    vggnet = vgg19(pretrained_path='vgg19-dcbb9e9d.pth', require_grad=False)
    vggnet = vggnet.to(device)
    vggnet.eval()

    # if args.auto_caption:
    #     input_dim = 640
    # else:
    #     input_dim = 512
    input_dim = 640
    text2Color = Text2Color(input_dim=input_dim).to(device)
    colorUNet = ColorUNet(bilinear=True).to(device)

    g_optim = optim.Adam(
        list(text2Color.parameters()) + list(colorUNet.parameters()),
        lr=args.lr,
        # betas=(0.9, 0.99),
        weight_decay=1e-5
        )
    # scheduler = torch.optim.lr_scheduler.StepLR(g_optim, step_size=100, gamma=0.1)
    lr = args.lr
    scheduler_f = partial(optim.lr_scheduler.CyclicLR,
                      base_lr=lr / 1000, max_lr=lr * 50,
                      step_size_up=4,
                      mode='triangular2',
                      cycle_momentum=False)
    scheduler = scheduler_f(g_optim)
    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0]) if not args.load_only_colorUNet else 0

        except ValueError:
            pass

        if not args.load_only_colorUNet:
            text2Color.load_state_dict(ckpt["text2Color"])
            g_optim.load_state_dict(ckpt["g_optim"])
            # scheduler.load_state_dict(ckpt["scheduler"])
        colorUNet.load_state_dict(ckpt["colorUNet"])

    # print(args.distributed)

    if args.distributed:
        text2Color = nn.parallel.DistributedDataParallel(
            text2Color,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        colorUNet = nn.parallel.DistributedDataParallel(
            colorUNet,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    size = 256
    transform = transforms.Compose(
        [
            transforms.Resize(288),
            # transforms.RandomResizedCrop(size, scale=(1, 1)),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(degrees=(0, 360))
        ]
    )

    datasets = []
    dataset = MultiResolutionDataset(transform, args.size)
    datasets.append(dataset)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=12
    )

    train(
        args,
        loader,
        text2Color,
        colorUNet,
        vggnet,
        g_optim,
        device,
        scheduler
    )
