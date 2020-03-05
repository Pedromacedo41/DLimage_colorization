from torchvision import transforms
import pathlib
from PIL import Image

class ImageLoader:
  def __init__(self, root, transform=None):
    self._transform = transform
    self.images = []
    self.class_to_idx = {}
    path = pathlib.Path(root)
    for i in path.rglob('*'):
      if i.is_file():
        _class = i.parts[-2]
        _class_id = self.class_to_idx.get(_class, len(self.class_to_idx))
        self.class_to_idx[_class] = _class_id
        self.images.append((i, _class_id))
    self.classes = list(self.class_to_idx.keys())

  def __getitem__(self, idx):
    _path, _class = self.images[idx]
    im = Image.open(_path)
    if self._transform:
      im = self._transform(im)
    return (im, _class)

  

if __name__=='__main__':
  il = ImageLoader('images', transform=transforms.ToTensor())
  print(il.classes)
  print(il.class_to_idx)
  print(il.images)
  print(il[0])