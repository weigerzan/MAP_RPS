import os
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CocoVal100(Dataset):
    def __init__(self, data_root, target_res=512, transform=None):
        self.image_dir = os.path.join(data_root, "images")
        meta_path = os.path.join(data_root, "metadata.csv")
        self.samples = []
        with open(meta_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample["saved_filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = sample["caption"]
        metadata = {
            "image_id": sample["image_id"],
            "orig_filename": sample["orig_filename"],
            "saved_filename": sample["saved_filename"],
            "orig_width": int(sample["orig_width"]),
            "orig_height": int(sample["orig_height"]),
        }
        return {
            "image": image,
            "caption": caption,
            "metadata": metadata
        }


if __name__ == "__main__":
    dataset = CocoVal100(data_root="exp/coco_100_for_inversion_512")
    print(dataset[0])