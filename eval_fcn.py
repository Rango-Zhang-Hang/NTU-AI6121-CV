import os
from tqdm import *

import click
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2

from cycada.data.data_loader import dataset_obj
from cycada.data.data_loader import get_fcn_dataset
from cycada.models.models import get_model
from cycada.models.models import models
from cycada.util import to_tensor_raw

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]


def fmt_array(arr, fmt=','):
    strs = ['{:.3f}'.format(x) for x in arr]
    return fmt.join(strs)

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def remap_labels2rgb(tensor):
  out = np.ones((3,tensor.shape[1],tensor.shape[2]), dtype = np.uint8)
  for i in range(tensor.shape[1]):
    for j in range(tensor.shape[2]):
      a = tensor[0,i,j]
      out[0,i,j] = palette[a*3]
      out[1,i,j] = palette[a*3+1]
      out[2,i,j] = palette[a*3+2]  
  return out

def result_stats(hist):
    acc_overall = np.diag(hist).sum() / hist.sum() * 100
    acc_percls = np.diag(hist) / (hist.sum(1) + 1e-8) * 100
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8) * 100
    freq = hist.sum(1) / hist.sum()
    fwIU = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc_overall, acc_percls, iu, fwIU


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--dataset', default='cityscapes',
              type=click.Choice(dataset_obj.keys()))
@click.option('--datadir', default='',
        type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--gpu', default='0')
@click.option('--num_cls', default=19)
def main(path, dataset, datadir, model, gpu, num_cls):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    net = get_model(model, num_cls=num_cls, weights_init=path)
    net.eval()
    ds = get_fcn_dataset(dataset, datadir, split='val', 
            transform=net.transform, target_transform=to_tensor_raw)
    classes = ds.classes
    loader = torch.utils.data.DataLoader(ds, num_workers=8)

    intersections = np.zeros(num_cls)
    unions = np.zeros(num_cls)

    errs = []
    hist = np.zeros((num_cls, num_cls))
    if len(loader) == 0:
        print('Empty data loader')
        return
    iterations = tqdm(enumerate(loader))
    i = 1
    for im_i, (im, label) in iterations:
        im = Variable(im.cuda())
        score = net(im).data
        _, preds = torch.max(score, 1)
        hist += fast_hist(label.numpy().flatten(),
                preds.cpu().numpy().flatten(),                                                                
                num_cls)        
        a = label.numpy()
        b = preds.cpu().detach().numpy()
        #print(a.shape())
        #torch.set_printoptions(profile="full")
        #print(b.to_dense())
        #pred_color = color_label(b, palette, with_void=False)
        #plt.figure()
        pred = remap_labels2rgb(b)
        pred_img = torch.from_numpy(pred).permute(1,2,0).numpy()
        red = pred_img[:,:,2]
        blue = pred_img[:,:,0]
        pred_img[:,:,0] = red
        pred_img[:,:,2] = blue
        #plt.imshow(pred_img)
        path = 'result/pred_' + str(i) + '.png'
        cv2.imwrite(path, pred_img)
        #plt.savefig(path)
        #plt.show()
        i += 1

        print('====================')

        acc_overall, acc_percls, iu, fwIU = result_stats(hist)
        iterations.set_postfix({'mIoU':' {:0.2f}  fwIoU: {:0.2f} pixel acc: {:0.2f} per cls acc: {:0.2f}'.format(
            np.nanmean(iu), fwIU, acc_overall, np.nanmean(acc_percls))})
    print()
    print(','.join(classes))
    print(fmt_array(iu))
    print(np.nanmean(iu), fwIU, acc_overall, np.nanmean(acc_percls))
    print()
    print('Errors:', errs)

if __name__ == '__main__':
    main()
