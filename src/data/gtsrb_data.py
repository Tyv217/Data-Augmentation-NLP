import pytorch_lightning as pl 
from torch.nn.functional import cross_entropy, softmax
import datetime, os, random 
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd, numpy as np
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import DataLoader

class GTSRBData(pl.LightningDataModule):
    def __init__(self, batch_size: int = 128):
      super().__init__()
      self.batch_size = batch_size
      self.convert_tensor = transforms.ToTensor()
      train_df = pd.read_csv("/home/x/xty20/project/content/gtsrb-german-traffic-sign/Train.csv")
      test_df = pd.read_csv("/home/x/xty20/project/content/gtsrb-german-traffic-sign/Train.csv")
      self.train = self.process_df(train_df, set_norm = True)
      self.test_dataset = self.process_df(test_df)

    def set_norm_values(self, img_tensors):
      channels, _, _ = img_tensors[0].shape
      mean = torch.zeros(channels)
      std = torch.zeros(channels)
      for i in range(channels):
        for j in range(len(img_tensors)):
          mean[i] += img_tensors[j][i].mean()
          std[i] += img_tensors[j][i].std()
      mean = mean / len(img_tensors)
      std = std / len(img_tensors)
      self.mean = mean
      self.std = std

    def resize_and_normalize_images(self, df, set_norm):
      image = df['Path'].apply(lambda x: Image.open("/home/x/xty20/project/content/gtsrb-german-traffic-sign/" + x))

      image = image.apply(lambda x: x.resize((32,32)))

      df['img_tensor'] = image.apply(lambda x: self.convert_tensor(x))
      
      if set_norm:
        self.set_norm_values(df['img_tensor'])

      df['img_tensor'] = df['img_tensor'].apply(lambda x: transforms.Normalize(mean=self.mean, std=self.std)(x))

      df['Roi.X1'] = df['Roi.X1'] * 32 / df['Width']
      df['Roi.Y1'] = df['Roi.Y1'] * 32 / df['Height']
      df['Roi.X2'] = df['Roi.X2'] * 32 / df['Width']
      df['Roi.Y2'] = df['Roi.Y2'] * 32 / df['Height']

      df['Roi.X1'] = (np.rint(df['Roi.X1'])).astype(int)
      df['Roi.Y1'] = (np.rint(df['Roi.Y1'])).astype(int)
      df['Roi.X2'] = (np.rint(df['Roi.X2'])).astype(int)
      df['Roi.Y2'] = (np.rint(df['Roi.Y2'])).astype(int)

      return df

    def process_df(self, df, set_norm = False):
      df = self.resize_and_normalize_images(df, set_norm)
      images = df['img_tensor']
      boxes = zip(df['Roi.X1'], df['Roi.Y1'], df['Roi.X2'], df['Roi.Y2'])
      boxes = [torch.tensor(box).unsqueeze(0) for box in boxes]
      labels = [torch.tensor(id) for id in df['ClassId']]
      target = []
      for image, (box, label) in zip(images, zip(boxes, labels)):
        target.append((image, {"boxes": box, "labels": label}))
      return target

    def setup(self, stage: str):
      train_split = 0.9
      train_len = int(len(self.train) * 0.9)
      self.train_dataset = self.train[:train_len]
      self.train_dataset = self.train_dataset[:int(len(self.train_dataset) / 3)]
      # print(self.train_dataset[:10])
      self.val_dataset = self.train[train_len:]
      self.val_dataset = self.val_dataset[:int(len(self.val_dataset))]

    def train_dataloader(self):
      return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True)

    def val_dataloader(self):
      return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
      return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
      return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def teardown(self, stage: str):
      # Used to clean-up when the run is finished
      ...