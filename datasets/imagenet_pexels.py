import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def get_imagenet_pexels(data_path, split="train", preprocessing=True):

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    if not preprocessing:
        transform = transforms.Compose([t for t in transform.transforms if not isinstance(t, transforms.Normalize)])
    return PexelsDataset(data_path, transform=transform)


class PexelsDataset(Dataset):

    def __init__(self, path, transform=None):
        super(PexelsDataset, self).__init__()

        classes = {130: "flamingo", 949: 'strawberry'}
        self.transform = transform
        self.num_classes = 1000
        self.classes = [str(i) for i in range(1000)]
        self.class_names = self.classes
        self.preprocessing = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        img_paths = []
        labels = []
        for cl in classes:
            img_paths += glob.glob(path + f"/{classes[cl]}/*")
            labels += [cl] * len(glob.glob(path + f"/{classes[cl]}/*"))

        test_sample = path + '/test/test_flamingo.jpg'
        self.img_paths = img_paths + [test_sample]
        self.class_id = labels + [130]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.class_id[idx]

    def get_target(self, idx):
        return self.class_id[idx]
