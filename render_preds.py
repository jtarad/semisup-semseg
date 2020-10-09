import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
from data import sen2
from data import get_loader, get_data_path
from PIL import Image
import argparse
import os
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import ConcatDataset
from model.deeplabv2 import Res_Deeplab
from utils.metric import _confusion_matrix, _acc

input_size = (64,64)
batch_size = 100
WIDTH = 182
HEIGHT = 210
IM_SZ = 52
overlap = 6
OUT_DIR = './render/'
OFF_SET = 210 - HEIGHT
RESTORE_FROM = './checkpoints/sen2_64_fm_reg'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=RESTORE_FROM)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()

def get_dataloader(data_loader, data_path, input_size, partitions, purpose):
    imgs_list = []
    for partition in partitions:
        imgs_curr = data_loader(
            **{"dataset_root": data_path,
               "input_sz": input_size[0],
               "gt_k": 15,
               "split": partition,
               "purpose": purpose}  # return testing tuples, image and label
        )
        imgs_list.append(imgs_curr)

    return ConcatDataset(imgs_list)

def render(flat_preds, names, colour_map):
    if not os.path.exists('./rend'):
      os.makedirs('./rend')
    flat_preds = np.array(flat_preds.cpu())
    n, w, h = flat_preds.shape
    for i in range(n):
      img_rend = np.zeros((w,h,3), dtype=np.uint8)
      for c in range(15):
        img_rend[flat_preds[i,:,:] == c, :] = colour_map[c]
      img_rend = Image.fromarray(img_rend)
      img_rend.save('./rend/'+str(names[i])+'.png')

def main():

    args = get_arguments()
    gpu = args.gpu

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    data_loader = get_loader('sen2')
    data_path = get_data_path('sen2')
    dataset = get_dataloader(data_loader, data_path, input_size, ["labelled_train"], "train")
    dataloader = data.DataLoader(dataset,
                    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = Res_Deeplab(num_classes=15)
    model.cuda(gpu)

    model = torch.nn.DataParallel(model).cuda()
    saved_state_dict = torch.load(os.path.join(args.model_path, "best.pth"))
    model.load_state_dict(saved_state_dict)

    model.eval()

    dataset_size = len(dataset)
    print("dataset size : ", dataset_size)

    num_samples_r = 0
    #num_samples = 0
    num_batches = int(dataset_size / batch_size) + 1
    #samples_per_batch = batch_size * input_size[0] * input_size[1]
    #colour_map = [(np.random.rand(3) * 255.).astype(np.uint8) for _ in range(15)]
                  
    colour_map = [np.array([255, 254, 145],dtype=np.uint8), # culture d'été
                  np.array([236, 96, 42],dtype=np.uint8),  # culture d'hiver
                  np.array([69, 153, 43],dtype=np.uint8),  # foret feuilles caduques
                  np.array([17, 49, 8],dtype=np.uint8),  # foret feuilles persistentes
                  np.array([170, 169, 53],dtype=np.uint8),  # pelouse
                  np.array([107, 168, 130],dtype=np.uint8),  # lande ligneuse
                  np.array([236, 102, 247],dtype=np.uint8),  # batiments denses
                  np.array([243, 175, 250],dtype=np.uint8),  # batiments diffus
                  np.array([179, 35, 191],dtype=np.uint8),  # zones industrielles
                  np.array([115, 251, 253],dtype=np.uint8),  # surface route
                  np.array([245, 186, 65],dtype=np.uint8),  # surface minerale
                  np.array([0, 30, 245],dtype=np.uint8),  # eau 
                  np.array([169, 171, 249],dtype=np.uint8),  # prairie
                  np.array([77, 10, 5],dtype=np.uint8),  # verger
                  np.array([127, 243, 74],dtype=np.uint8)  # vigne
                  ]
                  

    preds_all = torch.zeros((batch_size * num_batches, IM_SZ, IM_SZ)).cuda()
    #flat_predss_all = torch.zeros((num_batches * samples_per_batch),
    #                             dtype=torch.uint8).cuda()
    #flat_labels_all = torch.zeros((num_batches * samples_per_batch),
    #                             dtype=torch.uint8).cuda()
    for b_i, batch in enumerate(dataloader):
        print("Loading data batch %s" % b_i)
        imgs = batch
        imgs = Variable(imgs).cuda(gpu)

        with torch.no_grad():
          x_outs_curr = F.interpolate(model(imgs), size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
        n, c, w, h = x_outs_curr.shape

        #actual_samples_curr = n * w * h
        #num_samples += actual_samples_curr
        #start_i = b_i * samples_per_batch
        start_r = b_i * batch_size
        num_samples_r += n
        preds_curr = torch.argmax(x_outs_curr, dim=1)

        preds_all[start_r:(start_r + n), :, :] = preds_curr[:,overlap:-overlap,overlap:-overlap]

        #flat_predss_all[start_i:(start_i + actual_samples_curr)] = preds_curr.view(-1)
        #flat_labels_all[start_i:(start_i + actual_samples_curr)] = labels.view(-1)

    preds_all = preds_all[:num_samples_r,:,:]

    #flat_predss_all = flat_predss_all[:num_samples]
    #flat_labels_all = flat_labels_all[:num_samples]

    #acc = _acc(flat_predss_all, flat_labels_all, c)
    #print(acc)
    preds_all = np.array(preds_all.cpu())
    img_preds = np.zeros((WIDTH*IM_SZ, HEIGHT*IM_SZ, 3), dtype=np.uint8)
    n = 0
    for i in range(WIDTH):
      for j in range(HEIGHT):

        im = preds_all[n,:,:]
        img_curr = np.zeros((IM_SZ,IM_SZ,3), dtype=np.uint8)
        for c in range(15):
          img_curr[im == c, :] = colour_map[c]

        start_w = i * IM_SZ
        end_w = (i + 1) * IM_SZ  
        start_h = j * IM_SZ
        end_h = (j + 1) * IM_SZ    
        img_preds[start_w:end_w, start_h:end_h, :] = img_curr
        n += 1
        if j+1 == HEIGHT:
          n += OFF_SET

    Image.fromarray(img_preds).save(os.path.join(args.out_dir,'rend_s4GAN.png'))

if __name__ == '__main__':
    main()