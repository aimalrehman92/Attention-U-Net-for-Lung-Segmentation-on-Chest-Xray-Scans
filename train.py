import os
import numpy as np
import pandas as pd
import datetime

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import cv2
from PIL import Image
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import albumentations as A

from attention_unet import AttU_Net
from utilities import XrayDataset
from utilities import distort_minibatch, separate_target_from_input

# --------------------------------------------------------------------- #

# path to the dataset and model

base_path = "/content/gdrive/MyDrive/Chest_XRay_Segmentation/Chest_Xray_data/Lung Segmentation/"

train_images_path = base_path + 'CXR_png/'
train_segments_path = base_path + 'masks/'
test_images_path =  base_path + 'test/'
prefixModelName = base_path + 'saved_models_CXR_Att_UNet_200/'
log_filename = base_path + "epoch_logs_CXR_Att_UNet_200.csv"


# training settings
n_epochs = 200
batch_size = 4
saveModelevery = 2
validateModelevery = 2
distort_after_every_epoch = 2


# particulars of the Adam optimizer
lr = 0.0002
b1 = 0.5
b2 = 0.999


# configuration flags
is_transform_train = True
is_transform_test = False
generate_epoch_log = True
shuffle_minibatch = True


# gpu settings
cuda = True if torch.cuda.is_available() else False
print("Torch.Cuda available: ", cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


# build dataloader
train_dataset = XrayDataset(root_dir=base_path, inference=False, train=True)
test_dataset = XrayDataset(root_dir=base_path, inference=False, train=False)

batch_size = 4

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# initiate the model
model = AttU_Net()
if cuda:
    model.to(device)


# initiate optimizer, loss function and data types
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
loss_function = nn.CrossEntropyLoss()


# ------------- Training Loop ------------- #

t0 = datetime.datetime.now() # starting time

for epoch in range(0, n_epochs):

    train_loss_total, test_loss_total = 0, 0

    print("Epoch : [{}/{}]".format(epoch+1, n_epochs))

    # Training Set
    print("Training...")
    for i, (images, masks) in enumerate(train_dataloader):

        if is_transform_train and (epoch+1)%distort_after_every_epoch != 0:

          images = images.detach().cpu().numpy()
          masks = masks.detach().cpu().numpy()

          cascaded = np.concatenate((images, masks), axis=1)
          cascaded = distort_minibatch(cascaded, batch_size)
          images, masks = separate_target_from_input(cascaded, batch_size, 3)

          del(cascaded)

          images = torch.from_numpy(images).float().to(device)
          masks = torch.from_numpy(masks).float().to(device)

        masks = torch.squeeze(masks, dim=1)

        images = Variable(images.type(FloatTensor))
        masks = Variable(masks.type(LongTensor))

        synth_logits = model(images)
        optimizer.zero_grad()

        train_loss = loss_function(synth_logits, masks) # plus any other loss
        train_loss.backward()
        optimizer.step()

        # for tensorboard
        train_loss_total += train_loss.item()

    # Print loss on console
    print("TRAINING FIGURES:")
    print("Cross Entropy Loss:", train_loss_total*batch_size/n_train_images)

    # Test Set
    print("Validating...")
    for i, (images, masks) in enumerate(test_dataloader):

        if is_transform_test and (epoch+1)%distort_after_every_epoch != 0:

          images = images.detach().cpu().numpy()
          masks = masks.detach().cpu().numpy()

          cascaded = np.concatenate((images, masks), axis=1)
          cascaded = distort_minibatch(cascaded, batch_size)
          images, masks = separate_target_from_input(cascaded, batch_size, 3)

          del(cascaded)

          images = torch.from_numpy(images).float().to(device)
          masks = torch.from_numpy(masks).float().to(device)

        masks = torch.squeeze(masks, dim=1)

        images = Variable(images.type(FloatTensor)) # wrap it into Variable
        masks = Variable(masks.type(LongTensor)) # wrap it into Variable

        synth_logits = model(images)

        test_loss = loss_function(synth_logits, masks) #+ if any other loss

        # for tensorboard
        test_loss_total += test_loss.item()

    # Print loss on console
    print("VALIDATION FIGURES:")
    print("Cross Entropy Loss:", test_loss_total*batch_size/n_test_images)

    # loss on tensorboard
    writer.add_scalars("xEntropy Loss", {'train': train_loss_total*batch_size/n_train_images,
                                        'valid': test_loss_total*batch_size/n_test_images}, epoch+1)

    # Save the Model
    if (epoch+1)%saveModelevery==0:
        state = {
            'epoch': epoch+1,
            'model': model.state_dict()}

        if not os.path.exists(prefixModelName):
            os.makedirs(prefixModelName)

        torch.save(state, prefixModelName+'/'+'%d.pt'%(epoch+1))
        print('save model: '+prefixModelName+'/'+'%d.pt'%(epoch+1))

    if generate_epoch_log:
        epoch_logs.loc[len(epoch_logs)] = [epoch+1,
                                            train_loss_total*batch_size/n_train_images,
                                            test_loss_total*batch_size/n_test_images,
                                           ]

# Save the epochs log
if generate_epoch_log:
    epoch_logs.to_csv(log_filename)
    print("Log file generated!")


tn = datetime.datetime.now()

print('\n ****** Training Finished Successfully! ****** \n')
print('--- Total Training Time: {} ---\n'.format(tn-t0))

writer.flush()
writer.close()
