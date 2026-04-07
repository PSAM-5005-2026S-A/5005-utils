import torchvision.transforms as T

from os import listdir, makedirs, system
from shutil import copy2

from torch import tensor

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator


class CustomizedDataset(ClassificationDataset):
  def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
    super().__init__(root, args, augment, prefix)

    # custom training transforms here
    train_transforms = T.Compose(
      [
        T.Resize((args.imgsz, args.imgsz)),
        T.RandomHorizontalFlip(p=args.fliplr),
        T.RandomVerticalFlip(p=args.flipud),
        T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
        T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v, saturation=args.hsv_s, hue=args.hsv_h),
        T.ToTensor(),
        T.Normalize(mean=tensor(0), std=tensor(1)),
        T.RandomErasing(p=args.erasing, inplace=True),
      ]
    )

    # custom validation transforms here
    val_transforms = T.Compose(
      [
        T.Resize((args.imgsz, args.imgsz)),
        T.ToTensor(),
        T.Normalize(mean=tensor(0), std=tensor(1)),
      ]
    )
    self.torch_transforms = train_transforms if augment else val_transforms


class CustomizedTrainer(ClassificationTrainer):
  def build_dataset(self, img_path: str, mode: str = "train", batch=None):
    return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)


class CustomizedValidator(ClassificationValidator):
  def build_dataset(self, img_path: str, mode: str = "train"):
    return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=self.args.split)


def prep_lfw_data():
  DATA_LFW_PATH = "data/image/lfw/cropped"
  system("wget -qO- https://github.com/PSAM-5020-2026S-A/5020-utils/releases/latest/download/lfw.tar.gz | tar xz")

  makedirs("lfw/train", exist_ok=True)
  makedirs("lfw/test", exist_ok=True)

  lfw_dirs = sorted(n for n in listdir(DATA_LFW_PATH))

  for d in lfw_dirs:
    makedirs(f"lfw/train/{d}", exist_ok=True)
    makedirs(f"lfw/test/{d}", exist_ok=True)
    fnames = [f for f in listdir(f"{DATA_LFW_PATH}/{d}")]
    train_cnt = int(len(fnames) * 0.8)

    for f in fnames[:train_cnt]:
      copy2(f"{DATA_LFW_PATH}/{d}/{f}", f"lfw/train/{d}/")
    for f in fnames[train_cnt:]:
      copy2(f"{DATA_LFW_PATH}/{d}/{f}", f"lfw/test/{d}/")

  system("rm -rf data/image")
