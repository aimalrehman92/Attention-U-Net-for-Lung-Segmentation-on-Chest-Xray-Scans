
import cv2
from PIL import Image

class XrayDataset(Dataset):
  def __init__(self, root_dir, inference=False, train=True, target_size=(512, 512)):

    self.inference = inference
    self.train = train
    self.root_dir = root_dir
    self.target_size = target_size

    if self.inference:

      self.image_folder = "test"
      self.image_path = os.path.join(self.root_dir, self.image_folder)
      self.images = list(os.listdir(self.image_path))

    else:

      self.image_folder = "CXR_png"
      self.mask_folder = "masks"

      self.image_path = os.path.join(self.root_dir, self.image_folder)
      self.mask_path = os.path.join(self.root_dir, self.mask_folder)

      self.masks = sorted([mask for mask in os.listdir(self.mask_path) if mask.endswith('_mask.png')])
      self.images = [mask[:-9] + ".png" for mask in self.masks]

      # Calculate the split index for training and test sets
      split_index = int(0.8 * len(self.images))

      if self.train:
        self.images = self.images[:split_index]
        self.masks = self.masks[:split_index]

      else:
        self.images = self.images[split_index:]
        self.masks = self.masks[split_index:]

  def __len__(self):
    return len(self.images)


  def __getitem__(self, idx):

    if self.inference:

      img_name = os.path.join(self.image_path, self.images[idx])

      # Load the image from the memory
      image = cv2.imread(img_name)  # Assuming images are in BGR format
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

      # Resize the image using OpenCV
      image = cv2.resize(image, self.target_size)
      image = np.transpose(image, (2, 0, 1))

      # Image name or patient information
      image_label = self.images[idx]

      return image, image_label

    else:

      img_name = os.path.join(self.image_path, self.images[idx])
      mask_name = os.path.join(self.mask_path, self.masks[idx])

      image = cv2.imread(img_name)  # Assuming images are in BGR format
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

      mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)  # Assuming masks are grayscale

      mask = mask / 255.0

      # Resize the image and mask using OpenCV
      image = cv2.resize(image, self.target_size)
      mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

      image = np.transpose(image, (2, 0, 1))
      mask = np.expand_dims(mask, axis=0)

      return image, mask


def distort_minibatch(slice_stack, batch_size):

    stacks_list = []

    for i in range(slice_stack.shape[0]): # loop over the minibatch

        # for each sample in the minibatch, apply the following:
        single_stack = slice_stack[i]
        single_stack = np.swapaxes(single_stack, 2, 0) # example: (17, 256, 256) -> (256, 256, 17)


        transform = A.Compose([
                              A.OneOf([
                              A.HorizontalFlip(p=1.0),
                              A.VerticalFlip(p=1.0),
                              A.RandomRotate90(p=0.5),
                              A.Rotate(p=0.5),
                              #A.GridDistortion(p=0.5)
                              ])])

        transformed_stack = transform(image=single_stack)
        transformed_stack = transformed_stack["image"]
        transformed_stack = np.swapaxes(transformed_stack, 2, 0) # example: (256, 256, 17) -> (17, 256, 256)
        #print(transformed_stack.shape)
        transformed_stack = np.expand_dims(transformed_stack, axis=0) # (1, 17, 256, 256)
        stacks_list.append(transformed_stack) # store in a list

    stacks_list = np.array(stacks_list)

    return stacks_list



def separate_target_from_input(slices_stack, batch_size=1, target_index=16):

    slices_stack = np.squeeze(slices_stack, axis=1)
    images_stack, segments_stack = [], []

    for i in range(slices_stack.shape[0]): # loop over the minibatch
        stack = slices_stack[i] # one stack of image+mask with dim: (4, 512, 512)
        segment, image = stack[target_index:, :, :], stack[:target_index, :, :]
        #segment = np.expand_dims(segment, axis=0)
        #image = np.expand_dims(image, axis=0)
        segments_stack.append(segment)
        images_stack.append(image)

    segments_stack = np.array(segments_stack)
    images_stack = np.array(images_stack)

    return images_stack, segments_stack


def find_best_model(dataframe, parameter):

  dataframe = dataframe.sort_values(by=parameter).reset_index().iloc[0]

  return int(dataframe['Epoch'])


def plot_mask(image, mask):

  fig, axs = plt.subplots(1, 2, figsize=(10, 5))

  axs[0].imshow(image, cmap='gray')
  axs[0].set_title('X-Ray Scan')
  axs[0].axis('off')

  axs[1].imshow(mask, cmap='Blues')
  axs[1].set_title('Predicted Segmentation')
  axs[1].axis('off')

  plt.suptitle(f"Subject:{subject[0]}")

  plt.show()
