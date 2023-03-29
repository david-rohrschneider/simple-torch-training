import os
import xml.etree.ElementTree as Et
import torch
import numpy as np
from torchvision.datasets import VisionDataset
from PIL import Image


class CustomVocDetection(VisionDataset):
    def __init__(self, root: str, transform=None):
        super().__init__(root, transform)
        self.ids = self.__get_ids()

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self.__get_annotation(image_id)

        image_path = os.path.join(self.root, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

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
        return len(self.ids)

    def __get_ids(self):
        ids = []
        for f in os.listdir(self.root):
            file_name = os.path.splitext(f)
            if file_name[1] == ".jpg":
                ids.append(file_name[0])
        return ids

    def __get_annotation(self, image_id):
        annotation_file = os.path.join(self.root, image_id + '.xml')
        tree = Et.parse(annotation_file)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            boxes.append([x1, y1, x2, y2])
            labels.append(label)

        return boxes, labels
