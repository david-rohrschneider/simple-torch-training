from torchvision import datasets
from typing import Callable, Optional, Any


class CustomClassification(datasets.ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None
    ):
        super().__init__(root, transform=transform)

    def __getitem__(self, index: int) -> Any:
        pass

    def __len__(self) -> int:
        pass


class CustomDetection(datasets.VisionDataset):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None
    ):
        super().__init__(root, transform=transform)

    def __getitem__(self, index: int) -> Any:
        pass

    def __len__(self) -> int:
        pass


class CustomSegmentation(datasets.VisionDataset):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None
    ):
        super().__init__(root, transform=transform)

    def __getitem__(self, index: int) -> Any:
        pass

    def __len__(self) -> int:
        pass
