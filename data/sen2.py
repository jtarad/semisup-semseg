from __future__ import print_function

import os
import os.path as osp

import cv2
import numpy as np
import scipy.io as sio
import torch
import torchvision.transforms as tvt
from PIL import Image
from torch.utils import data
from tqdm import tqdm

__all__ = ["Sen2"]

RENDER_DATA = False


class _Sen2(data.Dataset):
  """Base class
  This contains fields and methods common to all sen2 datasets

  """

  def __init__(self, dataset_root, input_sz, gt_k, split=None, purpose=None, preload=False):
    super(_Sen2, self).__init__()

    self.split = split
    self.purpose = purpose

    self.root = dataset_root

    self.partition_dir = "part30"

    self.gt_k = gt_k
    self.input_sz = input_sz

    self.preload = preload

    self.files = []
    self.images = []
    self.labels = []
    self.labels_test = []

    self._set_files()

    if self.preload:
      self._preload_data()

    cv2.setNumThreads(0)

  def _set_files(self):
    raise NotImplementedError()

  def _load_data(self, image_id):
    raise NotImplementedError()

  
  def _prepare_train(self, index, img):
    # Returns one pair only, i.e. without transformed second image.
    # Used for standard CNN training (baselines).
    # This returns gpu tensors.
    # label is passed in canonical [0 ... 181] indexing

    img = img.astype(np.float32)

    img1 = np.array(img)

    img1 = img1.astype(np.float32) / 10000.

    img1 = torch.from_numpy(img1).permute(2, 0, 1)

    return img1

  def _prepare_test(self, index, img, label):
    # This returns cpu tensors.
    #   Image: 3D with channels last, float32, in range [0, 1] (normally done
    #     by ToTensor).
    #   Label map: 2D, flat int64, [0 ... sef.gt_k - 1]
    # label is passed in canonical [0 ... 181] indexing

    assert (label is not None)

    assert (img.shape[:2] == label.shape)
    img = img.astype(np.float32)
    label = label.astype(np.int32)

    img = img.astype(np.float32) / 10000.

    img = torch.from_numpy(img).permute(2, 0, 1)

    label = self._filter_label(label)

    # dataloader must return tensors (conversion forced in their code anyway)
    return img, torch.from_numpy(label)
    
  def _preload_data(self):
    for image_id in tqdm(
      self.files, desc="Preloading...", leave=False, dynamic_ncols=True):
      image, label, test = self._load_data(image_id)
      self.images.append(image)
      self.labels.append(label)
      self.labels_test.append(test)

  def __getitem__(self, index):
    if self.preload:
      image, label, test = self.images[index], self.labels[index], self.labels_test[index]
    else:
      image_id = self.files[index]
      image, label, test = self._load_data(image_id)

    if self.purpose == "train":
      return self._prepare_train(index, image)
    elif self.purpose == "train_sup":
      return self._prepare_test(index, image, label)
    else:
      assert (self.purpose == "test")
      if test is not None:
        return self._prepare_test(index, image, test)
      else:
        return self.prepare_test(index, image, label)

  def __len__(self):
    return len(self.files)

  def _check_gt_k(self):
    raise NotImplementedError()

  def _filter_label(self, label):
    raise NotImplementedError()

  def _set_files(self):
    if self.split in ["labelled_train", "labelled_test"]:
      # deterministic order - important - so >1 dataloader actually meaningful
      file_list = osp.join(self.root, self.partition_dir, self.split + ".txt")
      file_list = tuple(open(file_list, "r"))
      file_list = [id_.rstrip() for id_ in file_list]
      self.files = file_list  # list of ids which may or may not have gt
    else:
      raise ValueError("Invalid split name: {}".format(self.split))

  def _load_data(self, image_id):
    image_path = osp.join(self.root, "imgs", image_id + ".mat")
    label_path = osp.join(self.root, "gt", image_id + ".mat")
    train_path = osp.join(self.root, "train", image_id + ".mat")
    test_path = osp.join(self.root, "test", image_id + ".mat")
    
    image = sio.loadmat(image_path)["img"]
    #assert (image.dtype == np.uint8)

    if os.path.exists(label_path):
      label = sio.loadmat(label_path)["gt"] #- 1
      #assert (label.dtype == np.int32)
      return image, label, None
    if (os.path.exists(train_path) & os.path.exists(test_path)):
      train = sio.loadmat(train_path)["gt"]
      test = sio.loadmat(test_path)["gt"]
      return image, train, test
    else:
      return image, None, None


class Sen2(_Sen2):
  def __init__(self, **kwargs):
    super(Sen2, self).__init__(**kwargs)

    #config = kwargs["config"]
    #self.use_coarse_labels = config.use_coarse_labels
    self._check_gt_k()

  def _check_gt_k(self):
    assert (self.gt_k == 15)

  def _filter_label(self, label):
    assert (label.max() < self.gt_k)
    return label
