import argparse
import os
import sys
import random
import timeit

import cv2
import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform
from torch.utils.data import ConcatDataset

from model.deeplabv2 import Res_Deeplab
#from model.deeplabv3p import Res_Deeplab 

from model.discriminator import s4GAN_discriminator
from utils.loss import CrossEntropy2d
from data.voc_dataset import VOCDataSet, VOCGTDataSet
from data import get_loader, get_data_path
from data.augmentations import *
from utils.metric import _confusion_matrix, _acc, _cohen_kappa_score
import matplotlib.pyplot as plt
from torchsummary import summary

start = timeit.default_timer()

DATA_DIRECTORY = './data/voc_dataset/'
DATA_LIST_PATH = './data/voc_list/train_aug.txt'
CHECKPOINT_DIR = './checkpoints/sen2_64_30/'

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
NUM_CLASSES = 21 # 21 for PASCAL-VOC / 60 for PASCAL-Context / 19 Cityscapes 
DATASET = 'pascal_voc' #pascal_voc or pascal_context 

SPLIT_ID = None

MODEL = 'DeepLab'
BATCH_SIZE = 8
NUM_STEPS = 40000
SAVE_PRED_EVERY = 1000

INPUT_SIZE = '321,321'
IGNORE_LABEL = -1 # 255 for PASCAL-VOC / -1 for PASCAL-Context / 250 for Cityscapes

RESTORE_FROM = './checkpoints/sen2_64_rendered2/best.pth'
RESTORE_FROM_D = './checkpoints/sen2_64_rendered2/best_D.pth'

LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
POWER = 0.9
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
NUM_WORKERS = 4
RANDOM_SEED = 1234

LAMBDA_FM = 0.1
LAMBDA_ST = 1.0
THRESHOLD_ST = 0.6 # 0.6 for PASCAL-VOC/Context / 0.7 for Cityscapes

LABELED_RATIO = None  #0.02 # 1/8 labeled data by default

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="dataset to be used")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--labeled-ratio", type=float, default=LABELED_RATIO,
                        help="ratio of the labeled data to full dataset")
    parser.add_argument("--split-id", type=str, default=SPLIT_ID,
                        help="split order id")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-fm", type=float, default=LAMBDA_FM,
                        help="lambda_fm for feature-matching loss.")
    parser.add_argument("--lambda-st", type=float, default=LAMBDA_ST,
                        help="lambda_st for self-training.")
    parser.add_argument("--threshold-st", type=float, default=THRESHOLD_ST,
                        help="threshold_st for the self-training threshold.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--ignore-label", type=float, default=IGNORE_LABEL,
                        help="label value to ignored for loss calculation")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of iterations.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=RESTORE_FROM_D,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                        help="Where to save checkpoints of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()

args = get_arguments()

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

def _eval(model, dataset, batch_size, input_size, num_batches, gpu, render_preds):

    torch.cuda.empty_cache()
    model.eval()
    print("EVAL")

    samples_per_batch = batch_size * input_size[0] * input_size[1]
    flat_predss_all = torch.zeros((num_batches * samples_per_batch),
                                 dtype=torch.uint8).cuda()
    flat_labels_all = torch.zeros((num_batches * samples_per_batch),
                                 dtype=torch.uint8).cuda()
    num_samples = 0
    colour_map = [(np.random.rand(3) * 255.).astype(np.uint8)
                  for _ in range(15)]
    for b_i, batch in enumerate(dataset):
        # print("Loading data batch %s" % b_i)
        imgs, labels = batch
        imgs = imgs.cuda()

        with torch.no_grad():
          x_outs_curr = F.interpolate(model(imgs), size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
        n, c, w, h = x_outs_curr.shape

        actual_samples_curr = n * w * h
        num_samples += actual_samples_curr
        start_i = b_i * samples_per_batch
        flat_preds_curr = torch.argmax(x_outs_curr, dim=1)

        flat_predss_all[start_i:(start_i + actual_samples_curr)] = flat_preds_curr.view(-1)
        flat_labels_all[start_i:(start_i + actual_samples_curr)] = labels.view(-1)

    flat_predss_all = flat_predss_all[:num_samples]
    flat_labels_all = flat_labels_all[:num_samples]

    acc = _acc(flat_predss_all, flat_labels_all, c)
    kappa_score = _cohen_kappa_score(flat_predss_all, flat_labels_all)
    cm = _confusion_matrix(flat_predss_all, flat_labels_all, c)

    model.train()
    torch.cuda.empty_cache()
    return acc, kappa_score, cm

def render(flat_preds, names, colour_map):
    out_dir = os.path.join(args.checkpoint_dir,'rend')
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
    flat_preds = np.array(flat_preds.cpu())
    n, w, h = flat_preds.shape
    for i in range(n):
      img_rend = np.zeros((w,h,3), dtype=np.uint8)
      for c in range(15):
        img_rend[flat_preds[i,:,:] == c, :] = colour_map[c]
      img_rend = Image.fromarray(img_rend)
      img_rend.save(os.path.join(out_dir, str(names[i])+'.png'))

def render_all(preds_all, colour_map):
    n = 0
    IM_SZ = 60
    WIDTH = 158
    HEIGHT = 182
    OFF_SET = 182 - HEIGHT
    preds_all = np.array(preds_all.cpu())  #.reshape((158*60, 182*60))
    img_preds = np.zeros((WIDTH*IM_SZ, HEIGHT*IM_SZ, 3), dtype=np.uint8)
    for i in range(WIDTH):
      for j in range(HEIGHT):
        im = preds_all[n,:,:]
        img_curr = np.zeros((60,60,3), dtype=np.uint8)
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

    Image.fromarray(img_preds).save('rend_s4GAN.png')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.PuRd):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def loss_calc(pred, label, weights, gpu):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d(ignore_label=args.ignore_label).cuda(gpu)  # Ignore label ??
    return criterion(pred, label, weights)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def compute_argmax_map(output):
    output = output.detach().cpu().numpy()
    output = output.transpose((1,2,0))
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
    output = torch.from_numpy(output).float()
    return output
     
def find_good_maps(D_outs, pred_all):
    count = 0
    for i in range(D_outs.size(0)):
        if D_outs[i] > args.threshold_st:
            count +=1

    if count > 0:
        print ('Above ST-Threshold : ', count, '/', args.batch_size)
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3))
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3))
        num_sel = 0 
        for j in range(D_outs.size(0)):
            if D_outs[j] > args.threshold_st:
                pred_sel[num_sel] = pred_all[j]
                label_sel[num_sel] = compute_argmax_map(pred_all[j])
                num_sel +=1
        return  pred_sel.cuda(), label_sel.cuda(), count  
    else:
        return 0, 0, count 

criterion = nn.BCELoss()

def main():
    print (args)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu

    # create network
    model = Res_Deeplab(num_classes=args.num_classes)
    
    model.train()
    model.cuda(args.gpu)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    summary(model,(110,64,64))

    # load pretrained parameters
    saved_state_dict = torch.load(args.restore_from)

    #new_params = model.state_dict().copy()
    #for name, param in new_params.items():
    #    if name in saved_state_dict and param.size() == saved_state_dict[name].size():
    #        new_params[name].copy_(saved_state_dict[name])
    #model.load_state_dict(saved_state_dict)

    # init D
    model_D = s4GAN_discriminator(num_classes=args.num_classes, dataset=args.dataset)

    fig, axarr = plt.subplots(6, sharex=False, figsize=(20, 20))

    #p_class = [0.0355738,  0.00141609, 0.48528844, 0.07337901, 0.0388637,  0.1026877,
    #0.00271774, 0.18383373, 0.0359457,  0.,         0.,         0.02593297,
    #0.01436112, 0.,         0.]  # small

    p_class = [7.25703402e-02, 1.57180553e-01, 1.81395714e-01, 2.15331438e-01,
    8.59744781e-02, 6.45834114e-02, 2.08535688e-03, 2.95754679e-02,
    2.30909954e-02, 1.18364523e-03, 5.40670110e-04, 4.34120229e-02,
    1.03664125e-01, 2.07385748e-03, 1.73379244e-02]

    #p_class = [0.0688357,  0.16084504, 0.17536666, 0.19976606, 0.12290404, 0.06232464,
    # 0.00192151, 0.02954287, 0.02216445, 0.0013851,  0.00159172, 0.03166692,
    # 0.1009182,  0.00169153, 0.01907556]  # 50%
 
    weights_t = (1/torch.log(1.02 + torch.tensor(p_class))).cuda()

    model_D = torch.nn.DataParallel(model_D).cuda()

    #if args.restore_from_D is not None:
        #model_D.load_state_dict(torch.load(args.restore_from_D))

    cudnn.benchmark = True    

    model_D.train()
    model_D.cuda(args.gpu)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        #os.makedirs(os.path.join(args.checkpoint_dir,'rend'))

    if args.dataset == 'pascal_voc':    
        train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
        #train_gt_dataset = VOCGTDataSet(args.data_dir, args.data_list, crop_size=input_size,
                        #scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    elif args.dataset == 'pascal_context':
        input_transform = transform.Compose([transform.ToTensor(),
            transform.Normalize([.406, .456, .485], [.229, .224, .225])])
        data_kwargs = {'transform': input_transform, 'base_size': 505, 'crop_size': 321}
        #train_dataset = get_segmentation_dataset('pcontext', split='train', mode='train', **data_kwargs)
        data_loader = get_loader('pascal_context')
        data_path = get_data_path('pascal_context') 
        train_dataset = data_loader(data_path, split='train', mode='train', **data_kwargs)
        #train_gt_dataset = data_loader(data_path, split='train', mode='train', **data_kwargs)
        
    elif args.dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        data_aug = Compose([RandomCrop_city((256, 512)), RandomHorizontallyFlip()])
        train_dataset = data_loader( data_path, is_transform=True, augmentations=data_aug) 
        #train_gt_dataset = data_loader( data_path, is_transform=True, augmentations=data_aug)

    elif args.dataset == 'sen2':
        data_loader = get_loader('sen2')
        data_path = get_data_path('sen2')
        train_dataset = get_dataloader(data_loader, data_path, input_size, ["labelled_train"], "train_sup")
        test_dataset = get_dataloader(data_loader, data_path, input_size, ["labelled_test"], "test")

    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)
    print ('train dataset size: ', train_dataset_size)
    print ('test dataset size: ', test_dataset_size)

    num_batches_train = int(train_dataset_size / args.batch_size) + 1
    last_batch_sz = train_dataset_size % args.batch_size
    
    num_batches_test = int(test_dataset_size / args.batch_size) + 1

    print('num batches train : ', num_batches_train)
    print('last batch size : ', last_batch_sz)

    if args.labeled_ratio is None:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        trainloader_gt = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        trainloader_remain_iter = iter(trainloader_remain)

        testloader = data.DataLoader(test_dataset,
                        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    else:
        partial_size = int(args.labeled_ratio * train_dataset_size)
        
        if args.split_id is not None:
            train_ids = pickle.load(open(args.split_id, 'rb'))
            print('loading train ids from {}'.format(args.split_id))
        else:
            train_ids = np.arange(train_dataset_size)
            np.random.shuffle(train_ids)
        
        pickle.dump(train_ids, open(os.path.join(args.checkpoint_dir, 'train_voc_split.pkl'), 'wb'))
        
        train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
        train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
        train_gt_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])

        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        trainloader_remain = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, sampler=train_remain_sampler, num_workers=4, pin_memory=True)
        trainloader_gt = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, sampler=train_gt_sampler, num_workers=4, pin_memory=True)

        trainloader_remain_iter = iter(trainloader_remain)

    trainloader_iter = iter(trainloader)
    trainloader_gt_iter = iter(trainloader_gt)

    # optimizer for segmentation network
    optimizer = optim.SGD(model.module.optim_parameters(args),
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D.zero_grad()

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    y_real_, y_fake_ = Variable(torch.ones(args.batch_size, 1).cuda()), Variable(torch.zeros(args.batch_size, 1).cuda())

    train_accs = []
    test_accs = []
    train_kscores = []
    test_kscores = []
    losses_ce = []
    losses_fm = []
    losses_S = []
    losses_D = []
    
    e_i = 0
    loss_ce_value = 0
    loss_D_value = 0
    loss_fm_value = 0
    loss_S_value = 0
    for i_iter in range(args.num_steps):

        #loss_ce_value = 0
        #loss_D_value = 0
        #loss_fm_value = 0
        #loss_S_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        # train Segmentation Network 
        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # training loss for labeled data only
        try:
            batch = next(trainloader_iter)
        except:
            print("end epoch %s" % e_i)
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        images, labels = batch
        images = Variable(images).cuda(args.gpu)
        pred = F.interpolate(model(images), size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
        loss_ce = loss_calc(pred, labels, weights_t, args.gpu) # Cross entropy loss for labeled data
        
        #training loss for remaining unlabeled data
        try:
            batch_remain = next(trainloader_remain_iter)
        except:
            trainloader_remain_iter = iter(trainloader_remain)
            batch_remain = next(trainloader_remain_iter)
        
        images_remain, labels_remain = batch_remain
        images_remain = Variable(images_remain).cuda(args.gpu)
        pred_remain = F.interpolate(model(images_remain), size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
        
        # concatenate the prediction with the input images
        images_remain = (images_remain-torch.min(images_remain))/(torch.max(images_remain)- torch.min(images_remain))
        #print (pred_remain.size(), images_remain.size())
        n, c, w, h = pred_remain.size()        
        mask = (labels_remain != args.ignore_label)
        pred_remain[mask.view(n, 1, w, h).repeat(1, c, 1, 1)] = 0       
        pred_cat = torch.cat((F.softmax(pred_remain, dim=1), images_remain), dim=1)
        
        D_out_z, D_out_y_pred = model_D(pred_cat) # predicts the D ouput 0-1 and feature map for FM-loss D_out_y_pred
  
        # find predicted segmentation maps above threshold 
        pred_sel, labels_sel, count = find_good_maps(D_out_z, pred_remain) 

        # training loss on above threshold segmentation predictions (Cross Entropy Loss)
        if count > 0 and i_iter > 0:
            loss_st = loss_calc(pred_sel, labels_sel, weights_t, args.gpu)
        else:
            loss_st = 0.0

        # Concatenates the input images and ground-truth maps for the Districrimator 'Real' input
        try:
            batch_gt = next(trainloader_gt_iter)
        except:
            trainloader_gt_iter = iter(trainloader_gt)
            batch_gt = next(trainloader_gt_iter)

        images_gt, labels_gt = batch_gt
        # Converts grounth truth segmentation into 'num_classes' segmentation maps. 
        D_gt_v = Variable(one_hot(labels_gt)).cuda(args.gpu)
                
        images_gt = images_gt.cuda()
        images_gt = (images_gt - torch.min(images_gt))/(torch.max(images)-torch.min(images))
            
        D_gt_v_cat = torch.cat((D_gt_v, images_gt), dim=1)
        D_out_z_gt , D_out_y_gt = model_D(D_gt_v_cat)  # D_out_y_gt
        
        # L1 loss for Feature Matching Loss
        loss_fm = torch.mean(torch.abs(torch.mean(D_out_y_gt, 0) - torch.mean(D_out_y_pred, 0)))
    
        if count > 0 and i_iter > 0: # if any good predictions found for self-training loss
            loss_S = loss_ce + args.lambda_st*loss_st +  args.lambda_fm*loss_fm
        else:
            loss_S = loss_ce +  args.lambda_fm*loss_fm

        loss_S.backward()
        loss_fm_value+= args.lambda_fm*loss_fm

        loss_ce_value += loss_ce.item()
        loss_S_value += loss_S.item()

        # train D
        for param in model_D.parameters():
            param.requires_grad = True

        # train with pred
        pred_cat = pred_cat.detach()  # detach does not allow the graddients to back propagate.
        
        D_out_z, _ = model_D(pred_cat)
        y_fake_ = Variable(torch.zeros(D_out_z.size(0), 1).cuda())
        loss_D_fake = criterion(D_out_z, y_fake_) 

        # train with gt
        D_out_z_gt , _ = model_D(D_gt_v_cat)
        y_real_ = Variable(torch.ones(D_out_z_gt.size(0), 1).cuda()) 
        loss_D_real = criterion(D_out_z_gt, y_real_)
        
        loss_D = (loss_D_fake + loss_D_real)/2.0
        loss_D.backward()
        loss_D_value += loss_D.item()

        optimizer.step()
        optimizer_D.step()

        print('iter = {0:8d}/{1:8d}, loss_ce = {2:.3f}, loss_fm = {3:.3f}, loss_S = {4:.3f}, loss_D = {5:.3f}'.format(i_iter, args.num_steps, loss_ce_value, loss_fm_value, loss_S_value, loss_D_value))
        
        # EVALUATION
        # -------------------------------------------------------------------------------
        
        if (i_iter % (num_batches_train-1) == 0) & (i_iter > 0):
          e_i += 1
          train_acc, train_kappa_score, train_cm = _eval(model, trainloader, args.batch_size, input_size, num_batches_train, args.gpu, render_preds=False)
          test_acc, test_kappa_score, test_cm = _eval(model, testloader, args.batch_size, input_size, num_batches_test, args.gpu, render_preds=False)

          print('train_acc = ',train_acc)
          print('test_acc = ',test_acc)

          cm_name_train = "confusion_matrix_%d.png" % e_i
          plot_confusion_matrix(cm=train_cm,classes=(range(args.num_classes)), normalize=True)   
          plt.savefig(os.path.join(args.checkpoint_dir, cm_name_train))

          avg_loss_ce = loss_ce_value / num_batches_train
          avg_loss_fm = loss_fm_value / num_batches_train
          avg_loss_S = loss_S_value / num_batches_train
          avg_loss_D = loss_D_value / num_batches_train

          loss_ce_value = 0
          loss_D_value = 0
          loss_fm_value = 0
          loss_S_value = 0

          train_accs.append(train_acc)
          test_accs.append(test_acc)
          train_kscores.append(train_kappa_score)
          test_kscores.append(test_kappa_score)
          losses_ce.append(avg_loss_ce)
          losses_fm.append(avg_loss_fm)
          losses_S.append(avg_loss_S)
          losses_D.append(avg_loss_D)

          axarr[0].clear()
          axarr[0].plot(train_accs, 'g')
          axarr[0].plot(test_accs, 'r')
          axarr[0].set_title("acc (best), top train : %f" % max(train_accs))

          axarr[1].clear()
          axarr[1].plot(train_kscores, 'g')
          axarr[1].plot(test_kscores, 'r')
          axarr[1].set_title("Cohen's kappa score (best) : %f" % max(train_kscores))

          axarr[2].clear()
          axarr[2].plot(losses_D)
          axarr[2].set_title("Loss D")

          axarr[3].clear()
          axarr[3].plot(losses_S)
          axarr[3].set_title("Loss S")

          axarr[4].clear()
          axarr[4].plot(losses_ce)
          axarr[4].set_title("Loss ce")
  
          axarr[5].clear()
          axarr[5].plot(losses_fm)
          axarr[5].set_title("Loss fm")
  
          fig.canvas.draw_idle()
          fig.savefig(os.path.join(args.checkpoint_dir, "plots.png"))

          if train_acc >= max(train_accs):
              print ('save model ...')
              torch.save(model.state_dict(),os.path.join(args.checkpoint_dir, 'best.pth'))
              torch.save(model_D.state_dict(),os.path.join(args.checkpoint_dir, 'best_D.pth'))

        if i_iter >= args.num_steps-1:
            print ('save model ...')
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir, 'latest.pth'))
            torch.save(model_D.state_dict(),os.path.join(args.checkpoint_dir, 'latest_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print ('saving checkpoint  ...')
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir, 'sen2_'+str(i_iter)+'.pth'))
            torch.save(model_D.state_dict(),os.path.join(args.checkpoint_dir, 'sen2_'+str(i_iter)+'_D.pth'))

    end = timeit.default_timer()
    print (end-start,'seconds')

if __name__ == '__main__':
    main()