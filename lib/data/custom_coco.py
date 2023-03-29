import torch
import numpy as np
from torchvision.datasets import VisionDataset, CocoDetection
from PIL import Image


class CustomCocoDetection(VisionDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.ann_file = f"{root}.json"
        self.coco = CocoDetection(root=self.root, annFile=self.ann_file)

    def __getitem__(self, index):
        image, target = self.coco[index]

        boxes = [b["bbox"] for b in target]
        labels = [b["category_id"] for b in target]

        if self.transform:
            bgr_image = np.flip(image, -1)
            transformed = self.transform(bgr_image)
            bgr_image = transformed["image"]
            rgb_image = np.flip(bgr_image, -1)
            image = Image.fromarray(rgb_image, "RGB")
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        # Convert boxes to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.int64)

        return image, boxes, labels

    def __len__(self):
        return len(self.coco)
