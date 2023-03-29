from torchvision import datasets

from configs import custom_dataset
from .custom_voc import CustomVocDetection
from .custom_coco import CustomCocoDetection


def get_custom_dataset_class(method: str, dataset_type: str):
    if method == "classification":
        if dataset_type == "imagefolder":
            return datasets.ImageFolder
        else:
            return custom_dataset.CustomClassification
    if method == "detection":
        if dataset_type == "coco":
            return CustomCocoDetection
        if dataset_type == "voc":
            return CustomVocDetection
        else:
            return custom_dataset.CustomDetection
    else:
        if dataset_type == "coco":
            return CustomCocoSegmentation
        if dataset_type == "voc":
            return CustomVocSegmentation
        else:
            return custom_dataset.CustomSegmentation
