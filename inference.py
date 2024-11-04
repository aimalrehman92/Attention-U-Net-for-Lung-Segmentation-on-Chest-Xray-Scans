
import os
import numpy as np
import pandas as pd
import datetime

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import cv2
from PIL import Image
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import albumentations as A

from attention_unet import AttU_Net
from utilities import XrayDataset


# directories
base_path = "/content/gdrive/MyDrive/Chest_XRay_Segmentation/Chest_Xray_data/Lung Segmentation/"
test_images_path =  base_path + 'test/'
inferences_path = base_path + 'inference/'
os.makedirs(inferences_path, exist_ok=True)

prefixModelName = base_path + 'saved_models_CXR_Att_UNet_200/'
model_file = prefixModelName + '50.pt'


# prepare the dataloader
test_dataset = XrayDataset(root_dir=base_path, inference=True, train=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0) # keep batch_size = 1


# load the model
model = AttU_Net()
if cuda:
    model.to(device)

checkpoint = torch.load(prefixModelName+model_file)
model.load_state_dict(checkpoint['model']) # # load model paramters stored as OrderedDict
model.eval()


# ------------ Inference Set ------------ #

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print("Processing images...")

for i, (image, image_label) in enumerate(test_dataloader):

  image = Variable(image.type(FloatTensor)) # wrap it into Variable

  # Make the inference
  synth_logits = model(image)
  synth_mask = F.softmax(synth_logits, dim=1)
  synth_mask = torch.argmax(synth_mask, dim=1)
  synth_mask = torch.squeeze(synth_mask, dim=0)

  synth_mask = synth_mask.cpu().numpy()

  # Loop through each image in the batch
  for j in range(synth_mask.shape[0]):
    # Save each image with its corresponding label
    image_path = os.path.join(output_dir, image_label[j])
    image_pil = Image.fromarray((synth_mask[j] * 255).astype('uint8'), mode='L')  # Convert each image separately
    image_pil.save(image_path)
    print(f'Saved {image_path}')
