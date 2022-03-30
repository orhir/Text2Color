import argparse
import os
from datetime import time
import random

import numpy as np
from skimage import color, io
import torchvision.datasets as dset
import torch
import torch.nn.functional as F
import clip
# import cv2
from PIL import Image
from models import ColorUNet
from text2color import Text2Color
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import CLIP_prefix_caption.predict
from transformers import GPT2Tokenizer
import skimage.color
import sklearn.neighbors as sknn


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class LookupEncode():
    '''Encode points using lookups'''

    def __init__(self, km_filepath=''):

        self.cc = np.load(km_filepath)
        self.offset = np.abs(np.amin(self.cc)) + 17  # add to get rid of negative numbers
        self.x_mult = 59  # differentiate x from y
        self.labels = {}
        for idx, (x, y) in enumerate(self.cc):
            x += self.offset
            x *= self.x_mult
            y += self.offset
            self.labels[x + y] = idx

    # returns bsz x 224 x 224 of bin labels (625 possible labels)
    def encode_points(self, pts_nd, grid_width=10):
        pts_flt = pts_nd.reshape((-1, 2))

        # round AB coordinates to nearest grid tick
        pgrid = np.round(pts_flt / grid_width) * grid_width

        # get single number by applying offsets
        pvals = pgrid + self.offset
        pvals = pvals[:, 0] * self.x_mult + pvals[:, 1]

        labels = np.zeros(pvals.shape, dtype='int32')
        # lookup in label index and assign values
        for k in self.labels:
            labels[pvals == k] = self.labels[k]

        return labels.reshape(pts_nd.shape[:-1])

    # return lab grid marks from probability distribution over bins
    def decode_points(self, pts_enc):
        return pts_enc.dot(self.cc)


# def error_metric(rec_img_rgb, orig_img_rgb):
#     eps = 0.000001
#
#     r, c, _ = rec_img_rgb.shape
#
#     rec_r, rec_g, rec_b = cv2.split(rec_img_rgb.astype('float32'))
#     orig_r, orig_g, orig_b = cv2.split(orig_img_rgb.astype('float32'))
#
#     Ir_r = (rec_r + rec_g + rec_b) / 3
#     Io_r = (orig_r + orig_g + orig_b) / 3
#
#     Ir_a = rec_b / (Ir_r + eps) - (rec_r + rec_g) / (2 * Ir_r + eps)
#     Ir_b = (rec_r - rec_g) / (Ir_r + eps)
#
#     Io_a = orig_b / (Io_r + eps) - (orig_r + orig_g) / (2 * Io_r + eps)
#     Io_b = (orig_r - orig_g) / (Io_r + eps)
#
#     error = np.sum((Ir_a - Io_a) ** 2) + np.sum((Ir_b - Io_b) ** 2)
#     return error / (r * c)  # remove this hardcoding later


# taken from https://github.com/superhans/colorfromlanguage
def accuracy(output, target, topk=1, mask=None):
    maxk = topk
    batch_size = 1
    # if mask is not None:
    #     batch_size = mask.sum().cpu().data[0]
    # else:
    #     batch_size = target.size(0)
    # _, pred = output.topk(maxk, 1, True, True)
    # pred = pred.t()
    # correct = pred.eq(target.view(1, -1).expand_as(pred)).float()
    correct = output.eq(target).float()

    if mask is not None:
        correct = correct * mask[None, :]

    res = []
    # correct_k = correct[:topk].view(-1).sum(0, keepdim=True)
    return correct.view(-1).sum(0) / correct.view(-1).shape[0] * (100.0 / batch_size)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def firstRepeat(sentence):
    words = sentence.split(' ')
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            return i + 1
    return len(words)


def Lab2RGB_out(img_lab):
    img_lab = img_lab.detach().cpu()
    img_l = img_lab[:, :1, :, :]
    img_ab = img_lab[:, 1:, :, :]
    # print(torch.max(img_l), torch.min(img_l))
    # print(torch.max(img_ab), torch.min(img_ab))
    img_l = img_l + 50
    pred_lab = torch.cat((img_l, img_ab), 1)[0, ...].numpy()
    # grid_lab = utils.make_grid(pred_lab, nrow=1).numpy().astype("float64")
    # print(grid_lab.shape)
    out = (np.clip(color.lab2rgb(pred_lab.transpose(1, 2, 0)), 0, 1) * 255).astype("uint8")
    return out


def RGB2Lab(inputs):
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


if __name__ == "__main__":
    device = "cuda"
    model_name = 'Imagenet_autoCaptioned__val'
    # ckpt_path = 'experiments/Color2Embed_noPreTrain_autoCaption/250000.pt'
    # ckpt_path = 'experiments/Color2Embed_noPreTrain_autoCaption/125000.pt'
    ckpt_path = 'experiments/ImageNet/250000.pt'
    test_dir_path = 'datasets/val_all/val_set/'
    colored_dir_path = 'datasets/val_all/val_set_recolored/'
    # test_dir_path = 'COCO_dset/val2014/'
    # colored_dir_path = 'COCO_dset/colored_val2014/'
    annoation_path = 'COCO_dset/annotations/captions_val2014.json'
    out_dir_path = 'results/gray2color/val2014/' + model_name
    imgsize = 256

    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", action='store_true')
    parser.add_argument("--imagenet", action='store_true')
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--description", type=str, default=None)
    parser.add_argument("--real_data", action='store_true')
    parser.add_argument("--metric", action='store_true')
    parser.add_argument("--auto_caption", action='store_true')
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()
    mkdirs(out_dir_path)

    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if args.real_data:
        dataset = dset.CocoCaptions(root=test_dir_path,
                                    annFile=annoation_path)
        out_dir_path = 'results/gray2color/val2014/real'
        mkdirs(out_dir_path)
        length = 5000
        for i in (pbar := tqdm(range(length))):
            idx = i
            out_img_path = os.path.join(out_dir_path, 'in%d.png' % idx)
            img = np.array(dataset[idx][0].convert("RGB"), 'uint8')
            io.imsave(out_img_path, img)
        exit(0)

    print("Loading Checkpoint")
    # if not args.auto_caption:
    #     clip_model, clip_preprocess = clip.load("ViT-B/32")
    # else:
    clip_model, clip_preprocess = clip.load("RN50x4", device=device, jit=False)

    # if args.auto_caption:
    input_dim = 640
    # else:
    #     input_dim = 512
    text2Color = Text2Color(input_dim=input_dim).to(device)
    text2Color.load_state_dict(ckpt["text2Color"])
    text2Color.eval()

    colorUNet = ColorUNet().to(device)
    colorUNet.load_state_dict(ckpt["colorUNet"])
    colorUNet.eval()
    print("Loading Done")

    imgs = []
    imgs_lab = []
    if args.imagenet:
        dataset = dset.ImageFolder(root=test_dir_path)
        colored_dataset = dset.ImageFolder(root=colored_dir_path)
    else:
        dataset = dset.CocoCaptions(root=test_dir_path, annFile=annoation_path)
        colored_dataset = dset.CocoCaptions(root=colored_dir_path, annFile=annoation_path)

    # length = (len(dataset) if args.metric else 5000) if (args.coco or args.imagenet) else 1
    length = len(dataset)

    val_loss = 0
    if args.auto_caption:
        prefix_length = 40
        caption_model = CLIP_prefix_caption.predict.ClipCaptionPrefix(prefix_length,
                                                                      clip_length=40,
                                                                      prefix_size=640,
                                                                      num_layers=8,
                                                                      mapping_type=CLIP_prefix_caption.predict.MappingType.Transformer)
        caption_model.load_state_dict(torch.load("CLIP_prefix_caption/transformer_weights.pt"))
        caption_model = caption_model.eval()
        caption_model = caption_model.to(device)
    # clip_size = 288 if args.auto_caption else 224
    clip_size = 288
    clip_resize = transforms.Resize((clip_size, clip_size))
    metric_resize = transforms.Resize((224, 224))
    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    real_image_clip_transform = torch.nn.Sequential(clip_resize)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # ACCURACY QUANTIZATION
    lookup_enc = LookupEncode('comparison_files/full_lab_grid_10.npy')
    num_classes = lookup_enc.cc.shape[0]

    for i in (pbar := tqdm(range(length))):
        # if args.metric:
        #     idx = i
        # else:
        #     idx = random.randint(0, len(colored_dataset)-1)
        idx = i
        with torch.no_grad():
            if args.coco:
                try:
                    img, img_caption = dataset[idx]
                    colored_img, _ = colored_dataset[idx]
                except FileNotFoundError:
                    print("{} Not Found".format(idx))
                    continue
            elif args.imagenet:
                img, _ = dataset[idx]
                colored_img, _ = colored_dataset[idx]

            else:
                img = Image.open(args.image_path)
                img_caption = [args.description]

            _img = img.convert("RGB")
            width, height = img.size
            img, img_lab = preprocessing(_img)
            img = img.to(device)
            img_lab = img_lab.to(device)

            if args.auto_caption:
                # encoded_real_image = clip_model.encode_image(clip_preprocess(_img.convert('L')).unsqueeze(0).to(device))
                encoded_real_image = clip_model.encode_image(clip_preprocess(colored_img).unsqueeze(0).to(device))
                prefix = encoded_real_image.to(device, dtype=torch.float32)
                prefix = prefix / prefix.norm(2, dim=1)[:, None]
                prefix_embed = caption_model.clip_project(prefix).reshape(1, prefix_length, -1)
                img_caption = CLIP_prefix_caption.predict.generate_beam(caption_model, tokenizer, embed=prefix_embed)[0]
                img_caption = img_caption.replace("black and white", "")
                if len(img_caption) > 50:
                    img_caption = " ".join(img_caption.split()[:firstRepeat(img_caption)])[:50]
                img_caption = [img_caption]
            z_captions = clip.tokenize(img_caption).to(device)

        with torch.no_grad():
            img1_L_resize = F.interpolate(img_lab[:, :1, :, :] / 50., size=(imgsize, imgsize), mode='bilinear',
                                          recompute_scale_factor=False, align_corners=False)

            clip_image = clip_preprocess(_img).unsqueeze(0).to(device)
            clip_score, _ = clip_model(clip_image, z_captions)
            best_indice = np.argmax(clip_score.cpu())
            z_caption = clip_model.encode_text(clip.tokenize(img_caption[best_indice]).to(device))
            out_img_path = os.path.join(out_dir_path, 'in%d_%s.png' % (idx, img_caption))
            color_vector = text2Color(z_caption)
            fake_ab_ = colorUNet((img1_L_resize, color_vector))
            fake_ab = F.interpolate(fake_ab_ * 110, size=(height, width), mode='bilinear',
                                    recompute_scale_factor=False,
                                    align_corners=False)

            fake_img = torch.cat((img_lab[:, :1, :, :], fake_ab), 1)
            img_ab = img_lab[:, 1:, :, :]
            fake_img = Lab2RGB_out(fake_img)
            if args.metric:
                source = torch.from_numpy(
                    lookup_enc.encode_points(metric_resize(fake_ab.squeeze()).permute(1, 2, 0).cpu()))
                target = torch.from_numpy(
                    lookup_enc.encode_points(metric_resize(img_ab.squeeze()).permute(1, 2, 0).cpu()))

                acc = accuracy(source, target)
                val_loss += acc
                pbar.set_description("Current Acc: {} | Total Accuracy: {}".format(acc, (val_loss / (idx + 1))))

            if not args.metric:
                io.imsave(out_img_path, fake_img)
            if not args.coco and not args.imagenet:
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                plt.imshow(_img)
                plt.title('Original')
                plt.axis('off')
                plt.subplot(2, 1, 2)
                plt.imshow(fake_img)
                plt.title(img_caption)
                plt.axis('off')
                plt.show()

    val_loss /= length
    print("ACC:", val_loss)
    # print('Average L2 Distance is: ', avg_l2)
    #             re_img, re_img_lab = preprocessing(fake_img)
    #             re_img = re_img.to(device)
    #             re_img_resize = F.interpolate(re_img / 255., size=(imgsize, imgsize), mode='bilinear',
    #                                           recompute_scale_factor=False, align_corners=False)
    #             re_color_vector = text2Color(re_img_resize)
    #
    #             cos_d = torch.cosine_similarity(color_vector, re_color_vector, dim=1)
    #         cos_d = cos_d / len(z_captions)
    #         cos_d_avg += cos_d
    #
    # print('Average Cosine Distance is: ', cos_d_avg / len(dataset))
