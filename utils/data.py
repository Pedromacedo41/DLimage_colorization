from torch.utils.data import Dataset
from torchvision import transforms
import pathlib
from PIL import Image
from skimage import color
import numpy as np


class ImageDataset(Dataset):
  def __init__(self, root, transform=None):
    self._transform = transform
    self.images = []
    self.class_to_idx = {}
    path = pathlib.Path(root)
    for i in path.rglob('*'):
      if i.is_file() and i.suffix in ('png', '.jpg', '.jpeg'):
        _class = i.parts[-2]
        _class_id = self.class_to_idx.get(_class, len(self.class_to_idx))
        self.class_to_idx[_class] = _class_id
        self.images.append((i, _class_id))
    self.classes = list(self.class_to_idx.keys())

  def __getitem__(self, idx):
    _path, _class = self.images[idx]
    rgb = Image.open(_path).convert("RGB")
    rgb = transforms.Resize((256,256), Image.BICUBIC)(rgb)
    rgb = np.array(rgb)
    Lab = color.rgb2lab(rgb).astype(np.float32).transpose(2,0,1)
    l = Lab[0,:,:][np.newaxis, ...]/100
    ab = (Lab[[1,2],:,:]+128)/256
    return (l, ab, _class)

  def __len__(self):
    return len(self.images)

  

if __name__=='__main__':
  il = ImageDataset('images', transform=np.array)
  print(il.classes)
  print(il.class_to_idx)
  print(il.images)
  print(il[0])
