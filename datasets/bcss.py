# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# import os
# import glob
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import cv2 as cv
# import re
# import logging

# logger = logging.getLogger(__name__)

# class BCSSTrainingDataset(Dataset):
#     CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
#     def __init__(self, img_root="../data/BCSS-WSSS/training", transform=None):
#         super(BCSSTrainingDataset, self).__init__()
#         self.img_root = img_root
#         self.transform = transform
#         self.img_paths = []
#         self.cls_labels = []
#         self.get_images_and_labels()

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, index):
#         img_path = self.img_paths[index]
#         cls_label = self.cls_labels[index]
#         try:
#             img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
#             if img is None or img.size == 0:
#                 logger.error(f"Failed to load image at {img_path}. Using dummy data.")
#                 img = np.zeros((224, 224, 3), dtype=np.uint8)  # Dummy image
#             if self.transform is not None:
#                 transformed = self.transform(image=img)
#                 img = transformed["image"]
#                 if img is None:
#                     logger.error(f"Transformation failed for {img_path}. Using dummy tensor.")
#                     img = torch.zeros((3, 224, 224))
#             else:
#                 img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0  # Convert to [C, H, W] tensor
#             cls_label = torch.tensor(cls_label, dtype=torch.float32)  # Ensure tensor
#             return os.path.basename(img_path), img, cls_label, torch.tensor(0)  # Dummy gt_label as tensor
#         except Exception as e:
#             logger.error(f"Error loading {img_path}: {str(e)}. Using dummy data.")
#             img = torch.zeros((3, 224, 224))
#             cls_label = torch.tensor([0, 0, 0, 0], dtype=torch.float32)  # Default cls_label
#             return os.path.basename(img_path), img, cls_label, torch.tensor(0)

#     def get_images_and_labels(self):
#         img_files = glob.glob(os.path.join(self.img_root, "*.png"))
#         logger.info(f"Found {len(img_files)} images in {self.img_root}")

#         for img_path in img_files:
#             try:
#                 img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
#                 if img is None or img.size == 0:
#                     logger.warning(f"Skipping invalid image at {img_path}")
#                     continue
#                 term_split = re.split("\[|\]", img_path)
#                 if len(term_split) < 2:
#                     logger.warning(f"Skipping {img_path}: no [cls_label] found in filename")
#                     continue
#                 cls_label = np.array([int(x) for x in term_split[1]])
#                 if len(cls_label) == len(self.CLASSES) - 1:  # Exclude background
#                     self.img_paths.append(img_path)
#                     self.cls_labels.append(cls_label)
#                 else:
#                     logger.warning(f"Skipping {img_path}: invalid cls_label length {len(cls_label)}")
#             except Exception as e:
#                 logger.error(f"Error processing {img_path}: {str(e)}")
#                 continue

#         logger.info(f"Loaded {len(self.img_paths)} valid training samples")

# class BCSSTestDataset(Dataset):
#     CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
#     PALETTE = {29: 0, 76: 1, 150: 2, 255: 3}  # Map dataset values to [0, 1, 2, 3], 255 as background
#     def __init__(self, img_root="../data/BCSS-WSSS/", split="test", transform=None):
#         assert split in ["test", "val"], "split must be one of [test, val]"
#         super(BCSSTestDataset, self).__init__()
#         self.img_root = img_root
#         self.split = split
#         self.transform = transform
#         self.img_paths = []
#         self.mask_paths = []
#         self.get_images_and_labels()

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, index):
#         img_path = self.img_paths[index]
#         mask_path = self.mask_paths[index]
#         try:
#             img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
#             if img is None or img.size == 0:
#                 logger.error(f"Failed to load image at {img_path}. Using dummy data.")
#                 img = np.zeros((224, 224, 3), dtype=np.uint8)  # Dummy image
#             mask = np.array(Image.open(mask_path).convert("L"))
#             if mask is None or mask.size == 0:
#                 logger.error(f"Failed to load mask at {mask_path}. Using dummy mask.")
#                 mask = np.zeros((224, 224), dtype=np.uint8)  # Dummy mask
#             # Map mask values to [0, 1, 2, 3, 4]
#             mask_mapped = np.zeros_like(mask, dtype=np.uint8)
#             for src_val, tgt_val in self.PALETTE.items():
#                 mask_mapped[mask == src_val] = tgt_val
#             mask_mapped[mask == 255] = 4  # Background
#             cls_label = np.array([0, 0, 0, 0])
#             unique_values = np.unique(mask_mapped)
#             foreground_values = unique_values[unique_values < len(self.CLASSES) - 1]  # [0, 1, 2, 3]
#             if len(foreground_values) > 0:
#                 cls_label[foreground_values] = 1
#             else:
#                 logger.warning(f"No foreground classes in mask {mask_path} after mapping. Setting cls_label to zeros.")
#             if self.transform is not None:
#                 transformed = self.transform(image=img, mask=mask_mapped)
#                 img = transformed["image"]
#                 mask = transformed["mask"]
#                 if img is None or mask is None:
#                     logger.error(f"Transformation failed for {img_path}. Using dummy tensors.")
#                     img = torch.zeros((3, 224, 224))
#                     mask = torch.zeros((224, 224), dtype=torch.long)
#             else:
#                 img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0  # Convert to [C, H, W] tensor
#                 mask = torch.from_numpy(mask_mapped).long()  # Convert to [H, W] tensor
#             cls_label = torch.tensor(cls_label, dtype=torch.float32)  # Ensure tensor [4]
#             return os.path.basename(img_path), img, cls_label, mask
#         except Exception as e:
#             logger.error(f"Error loading {img_path} or {mask_path}: {str(e)}. Using dummy data.")
#             img = torch.zeros((3, 224, 224))
#             mask = torch.zeros((224, 224), dtype=torch.long)
#             cls_label = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
#             return os.path.basename(img_path), img, cls_label, mask

#     def get_images_and_labels(self):
#         mask_paths = glob.glob(os.path.join(self.img_root, self.split, "mask", "*.png"))
#         logger.info(f"Found {len(mask_paths)} mask files in {os.path.join(self.img_root, self.split, 'mask')}")

#         for mask_path in mask_paths:
#             try:
#                 mask = np.array(Image.open(mask_path).convert("L"))
#                 if mask is None or mask.size == 0:
#                     logger.warning(f"Skipping invalid mask at {mask_path}")
#                     continue
#                 img_name = os.path.basename(mask_path)
#                 img_path = os.path.join(self.img_root, self.split, "img", img_name)
#                 if os.path.exists(img_path):
#                     img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
#                     if img is None or img.size == 0:
#                         logger.warning(f"Skipping {img_name}: invalid image")
#                         continue
#                     self.img_paths.append(img_path)
#                     self.mask_paths.append(mask_path)
#                 else:
#                     logger.warning(f"Image not found for {img_name} at {img_path}")
#             except Exception as e:
#                 logger.error(f"Error processing {mask_path}: {str(e)}")
#                 continue

#         logger.info(f"Loaded {len(self.img_paths)} valid {self.split} samples")

# class BCSSWSSSDataset(Dataset):
#     CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
#     def __init__(self, img_root="../data/BCSS-WSSS/training", mask_name="pseudo_label", transform=None):
#         super(BCSSWSSSDataset, self).__init__()
#         self.img_root = img_root
#         self.mask_name = mask_name
#         self.transform = transform
#         self.img_paths = []
#         self.mask_paths = []
#         self.cls_labels = []
#         self.get_images_and_labels()

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, index):
#         img_path = self.img_paths[index]
#         mask_path = self.mask_paths[index]
#         cls_label = self.cls_labels[index]
#         try:
#             img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
#             if img is None or img.size == 0:
#                 logger.error(f"Failed to load image at {img_path}. Using dummy data.")
#                 img = np.zeros((224, 224, 3), dtype=np.uint8)  # Dummy image
#             mask = np.array(Image.open(mask_path).convert("L"))
#             if mask is None or mask.size == 0:
#                 logger.error(f"Failed to load mask at {mask_path}. Using dummy mask.")
#                 mask = np.zeros((224, 224), dtype=np.uint8)  # Dummy mask
#             if self.transform is not None:
#                 transformed = self.transform(image=img, mask=mask)
#                 img = transformed["image"]
#                 mask = transformed["mask"]
#                 if img is None or mask is None:
#                     logger.error(f"Transformation failed for {img_path}. Using dummy tensors.")
#                     img = torch.zeros((3, 224, 224))
#                     mask = torch.zeros((1, 224, 224), dtype=torch.long)
#             else:
#                 img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0  # Convert to [C, H, W] tensor
#                 mask = torch.from_numpy(mask).unsqueeze(0).long()  # Convert to [1, H, W] tensor
#             cls_label = torch.tensor(cls_label, dtype=torch.float32)  # Ensure tensor
#             return os.path.basename(img_path), img, cls_label, mask
#         except Exception as e:
#             logger.error(f"Error loading {img_path} or {mask_path}: {str(e)}. Using dummy data.")
#             img = torch.zeros((3, 224, 224))
#             mask = torch.zeros((1, 224, 224), dtype=torch.long)
#             cls_label = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
#             return os.path.basename(img_path), img, cls_label, mask

#     def get_images_and_labels(self, img_root, mask_name):
#         img_paths = glob.glob(os.path.join(img_root, "img", "*.png"))
#         logger.info(f"Found {len(img_paths)} image files in {os.path.join(img_root, 'img')}")

#         for img_path in img_paths:
#             try:
#                 img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
#                 if img is None or img.size == 0:
#                     logger.warning(f"Skipping invalid image at {img_path}")
#                     continue
#                 img_name = os.path.basename(img_path)
#                 mask_path = os.path.join(img_root, mask_name, img_name)
#                 if os.path.exists(mask_path):
#                     mask = np.array(Image.open(mask_path).convert("L"))
#                     if mask is None or mask.size == 0:
#                         logger.warning(f"Skipping {img_name}: invalid mask")
#                         continue
#                     term_split = re.split("\[|\]", img_path)
#                     if len(term_split) < 2:
#                         logger.warning(f"Skipping {img_path}: no [cls_label] found in filename")
#                         continue
#                     cls_label = np.array([int(x) for x in term_split[1]])
#                     if len(cls_label) == len(self.CLASSES) - 1:  # Exclude background
#                         self.img_paths.append(img_path)
#                         self.mask_paths.append(mask_path)
#                         self.cls_labels.append(cls_label)
#                     else:
#                         logger.warning(f"Skipping {img_path}: invalid cls_label length {len(cls_label)}")
#                 else:
#                     logger.warning(f"Mask not found for {img_name} at {mask_path}")
#             except Exception as e:
#                 logger.error(f"Error processing {img_path}: {str(e)}")
#                 continue

#         logger.info(f"Loaded {len(self.img_paths)} valid WSSS samples")
        
        
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
import re
import logging

logger = logging.getLogger(__name__)

class BCSSTrainingDataset(Dataset):
    CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
    def __init__(self, img_root="../data/BCSS-WSSS/training", transform=None):
        super(BCSSTrainingDataset, self).__init__()
        self.img_root = img_root
        self.transform = transform
        self.img_paths = []
        self.cls_labels = []
        self.get_images_and_labels()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        cls_label = self.cls_labels[index]
        try:
            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
            if img is None or img.size == 0:
                logger.error(f"Failed to load image at {img_path}. Using dummy data.")
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                cls_label_tensor = torch.tensor(cls_label, dtype=torch.float32)
                h, w = 224, 224
                labels = torch.zeros(h, w, dtype=torch.long)
                active_classes = cls_label_tensor > 0
                if torch.any(active_classes):
                    num_active = torch.sum(active_classes).item()
                    section_size = h // num_active if num_active > 0 else h
                    label_map = torch.zeros(h, w)
                    active_indices = torch.where(active_classes)[0]
                    for idx, c in enumerate(active_indices):
                        if c >= len(self.CLASSES) - 1:
                            logger.warning(f"Index {c} out of bounds for num_classes {len(self.CLASSES) - 1}. Skipping.")
                            continue
                        start = idx * section_size
                        end = (idx + 1) * section_size if idx < num_active - 1 else h
                        label_map[start:end, :] = c.item()
                    labels = label_map.long()
                return os.path.basename(img_path), img, cls_label_tensor, labels

            if self.transform is not None:
                transformed = self.transform(image=img)
                img = transformed["image"]
                if img is None:
                    logger.error(f"Transformation failed for {img_path}. Using dummy tensor.")
                    img = torch.zeros((3, 224, 224))
            else:
                img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0

            cls_label_tensor = torch.tensor(cls_label, dtype=torch.float32)
            h, w = 224, 224
            labels = torch.zeros(h, w, dtype=torch.long)
            active_classes = cls_label_tensor > 0
            if torch.any(active_classes):
                num_active = torch.sum(active_classes).item()
                section_size = h // num_active if num_active > 0 else h
                label_map = torch.zeros(h, w)
                active_indices = torch.where(active_classes)[0]
                for idx, c in enumerate(active_indices):
                    if c >= len(self.CLASSES) - 1:
                        logger.warning(f"Index {c} out of bounds for num_classes {len(self.CLASSES) - 1}. Skipping.")
                        continue
                    start = idx * section_size
                    end = (idx + 1) * section_size if idx < num_active - 1 else h
                    label_map[start:end, :] = c.item()
                labels = label_map.long()

            return os.path.basename(img_path), img, cls_label_tensor, labels
        except Exception as e:
            logger.error(f"Error loading {img_path}: {str(e)}. Using dummy image but keeping original cls_label.")
            img = torch.zeros((3, 224, 224))
            cls_label_tensor = torch.tensor(cls_label, dtype=torch.float32)
            h, w = 224, 224
            labels = torch.zeros(h, w, dtype=torch.long)
            active_classes = cls_label_tensor > 0
            if torch.any(active_classes):
                num_active = torch.sum(active_classes).item()
                section_size = h // num_active if num_active > 0 else h
                label_map = torch.zeros(h, w)
                active_indices = torch.where(active_classes)[0]
                for idx, c in enumerate(active_indices):
                    if c >= len(self.CLASSES) - 1:
                        logger.warning(f"Index {c} out of bounds for num_classes {len(self.CLASSES) - 1}. Skipping.")
                        continue
                    start = idx * section_size
                    end = (idx + 1) * section_size if idx < num_active - 1 else h
                    label_map[start:end, :] = c.item()
                labels = label_map.long()
            return os.path.basename(img_path), img, cls_label_tensor, labels

    def get_images_and_labels(self):
        img_files = glob.glob(os.path.join(self.img_root, "*.png"))
        logger.info(f"Found {len(img_files)} images in {self.img_root}")

        for img_path in img_files:
            try:
                img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
                if img is None or img.size == 0:
                    logger.warning(f"Skipping invalid image at {img_path}")
                    continue
                term_split = re.split("\[|\]", os.path.basename(img_path))
                if len(term_split) < 2:
                    logger.warning(f"Skipping {img_path}: no [cls_label] found in filename")
                    continue
                cls_label_str = term_split[1]
                cls_label = np.array([int(x) for x in cls_label_str])
                if len(cls_label) != len(self.CLASSES) - 1:
                    logger.warning(f"Skipping {img_path}: invalid cls_label length {len(cls_label)}")
                    continue
                if not all(0 <= x <= 1 for x in cls_label):
                    logger.warning(f"Skipping {img_path}: invalid cls_label values {cls_label}")
                    continue
                self.img_paths.append(img_path)
                self.cls_labels.append(cls_label)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue

        logger.info(f"Loaded {len(self.img_paths)} valid training samples")

class BCSSTestDataset(Dataset):
    CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
    PALETTE = {29: 0, 76: 1, 150: 2, 255: 3}
    def __init__(self, img_root="../data/BCSS-WSSS/", split="test", transform=None):
        assert split in ["test", "val"], "split must be one of [test, val]"
        super(BCSSTestDataset, self).__init__()
        self.img_root = img_root
        self.split = split
        self.transform = transform
        self.img_paths = []
        self.mask_paths = []
        self.get_images_and_labels()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        try:
            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
            if img is None or img.size == 0:
                logger.error(f"Failed to load image at {img_path}. Using dummy data.")
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                mask = np.zeros((224, 224), dtype=np.uint8)
                cls_label = np.array([0, 0, 0, 0])
            else:
                mask = np.array(Image.open(mask_path).convert("L"))
                if mask is None or mask.size == 0:
                    logger.error(f"Failed to load mask at {mask_path}. Using dummy mask.")
                    mask = np.zeros((224, 224), dtype=np.uint8)
                    cls_label = np.array([0, 0, 0, 0])
                else:
                    mask_mapped = np.zeros_like(mask, dtype=np.uint8)
                    for src_val, tgt_val in self.PALETTE.items():
                        mask_mapped[mask == src_val] = tgt_val
                    mask_mapped[mask == 255] = 4
                    cls_label = np.array([0, 0, 0, 0])
                    unique_values = np.unique(mask_mapped)
                    foreground_values = unique_values[unique_values < len(self.CLASSES) - 1]
                    if len(foreground_values) > 0:
                        cls_label[foreground_values] = 1
                    else:
                        logger.warning(f"No foreground classes in mask {mask_path} after mapping. Setting cls_label to zeros.")
                    mask = mask_mapped

            if self.transform is not None:
                transformed = self.transform(image=img, mask=mask)
                img = transformed["image"]
                mask = transformed["mask"]
                if img is None or mask is None:
                    logger.error(f"Transformation failed for {img_path}. Using dummy tensors.")
                    img = torch.zeros((3, 224, 224))
                    mask = torch.zeros((224, 224), dtype=torch.long)
            else:
                img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
                mask = torch.from_numpy(mask).long()

            cls_label = torch.tensor(cls_label, dtype=torch.float32)
            return os.path.basename(img_path), img, cls_label, mask
        except Exception as e:
            logger.error(f"Error loading {img_path} or {mask_path}: {str(e)}. Using dummy data.")
            img = torch.zeros((3, 224, 224))
            mask = torch.zeros((224, 224), dtype=torch.long)
            cls_label = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            return os.path.basename(img_path), img, cls_label, mask

    def get_images_and_labels(self):
        mask_paths = glob.glob(os.path.join(self.img_root, self.split, "mask", "*.png"))
        logger.info(f"Found {len(mask_paths)} mask files in {os.path.join(self.img_root, self.split, 'mask')}")

        for mask_path in mask_paths:
            try:
                mask = np.array(Image.open(mask_path).convert("L"))
                if mask is None or mask.size == 0:
                    logger.warning(f"Skipping invalid mask at {mask_path}")
                    continue
                img_name = os.path.basename(mask_path)
                img_path = os.path.join(self.img_root, self.split, "img", img_name)
                if os.path.exists(img_path):
                    img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
                    if img is None or img.size == 0:
                        logger.warning(f"Skipping {img_name}: invalid image")
                        continue
                    self.img_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                else:
                    logger.warning(f"Image not found for {img_name} at {img_path}")
            except Exception as e:
                logger.error(f"Error processing {mask_path}: {str(e)}")
                continue

        logger.info(f"Loaded {len(self.img_paths)} valid {self.split} samples")

class BCSSWSSSDataset(Dataset):
    CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
    def __init__(self, img_root="../data/BCSS-WSSS/training", mask_name="pseudo_label", transform=None):
        super(BCSSWSSSDataset, self).__init__()
        self.img_root = img_root
        self.mask_name = mask_name
        self.transform = transform
        self.img_paths = []
        self.mask_paths = []
        self.cls_labels = []
        self.get_images_and_labels()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        cls_label = self.cls_labels[index]
        try:
            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
            if img is None or img.size == 0:
                logger.error(f"Failed to load image at {img_path}. Using dummy data.")
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                mask = np.zeros((224, 224), dtype=np.uint8)
                active_classes = np.where(cls_label == 1)[0]
                if len(active_classes) > 0:
                    section_size = 224 // len(active_classes)
                    for idx, cls_idx in enumerate(active_classes):
                        if cls_idx >= len(self.CLASSES) - 1:
                            logger.warning(f"Index {cls_idx} out of bounds for num_classes {len(self.CLASSES) - 1}. Skipping.")
                            continue
                        mask[idx * section_size : (idx + 1) * section_size, :] = cls_idx + 1
            else:
                mask = np.array(Image.open(mask_path).convert("L"))
                if mask is None or mask.size == 0:
                    logger.error(f"Failed to load mask at {mask_path}. Deriving simple mask from cls_label.")
                    mask = np.zeros((224, 224), dtype=np.uint8)
                    active_classes = np.where(cls_label == 1)[0]
                    if len(active_classes) > 0:
                        section_size = 224 // len(active_classes)
                        for idx, cls_idx in enumerate(active_classes):
                            if cls_idx >= len(self.CLASSES) - 1:
                                logger.warning(f"Index {cls_idx} out of bounds for num_classes {len(self.CLASSES) - 1}. Skipping.")
                                continue
                            mask[idx * section_size : (idx + 1) * section_size, :] = cls_idx + 1

            if self.transform is not None:
                transformed = self.transform(image=img, mask=mask)
                img = transformed["image"]
                mask = transformed["mask"]
                if img is None or mask is None:
                    logger.error(f"Transformation failed for {img_path}. Using dummy tensors.")
                    img = torch.zeros((3, 224, 224))
                    mask = torch.zeros((1, 224, 224), dtype=torch.long)
            else:
                img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
                mask = torch.from_numpy(mask).unsqueeze(0).long()

            cls_label = torch.tensor(cls_label, dtype=torch.float32)
            return os.path.basename(img_path), img, cls_label, mask
        except Exception as e:
            logger.error(f"Error loading {img_path} or {mask_path}: {str(e)}. Using dummy image and deriving mask from cls_label.")
            img = torch.zeros((3, 224, 224))
            cls_label = torch.tensor(self.cls_labels[index], dtype=torch.float32)
            mask = torch.zeros((1, 224, 224), dtype=torch.long)
            active_classes = torch.where(cls_label == 1)[0]
            if len(active_classes) > 0:
                section_size = 224 // len(active_classes)
                mask_np = mask.squeeze(0).cpu().numpy()
                for idx, cls_idx in enumerate(active_classes):
                    if cls_idx >= len(self.CLASSES) - 1:
                        logger.warning(f"Index {cls_idx} out of bounds for num_classes {len(self.CLASSES) - 1}. Skipping.")
                        continue
                    mask_np[idx * section_size : (idx + 1) * section_size, :] = cls_idx.item() + 1
                mask = torch.from_numpy(mask_np).unsqueeze(0).long()
            return os.path.basename(img_path), img, cls_label, mask


    def get_images_and_labels(self):
        img_paths = glob.glob(os.path.join(self.img_root, "img", "*.png"))
        logger.info(f"Found {len(img_paths)} image files in {os.path.join(self.img_root, 'img')}")

        for img_path in img_paths:
            try:
                img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
                if img is None or img.size == 0:
                    logger.warning(f"Skipping invalid image at {img_path}")
                    continue
                img_name = os.path.basename(img_path)
                mask_path = os.path.join(self.img_root, self.mask_name, img_name)
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path).convert("L"))
                    if mask is None or mask.size == 0:
                        logger.warning(f"Skipping {img_name}: invalid mask")
                        continue
                    term_split = re.split("\[|\]", img_name)
                    if len(term_split) < 2:
                        logger.warning(f"Skipping {img_path}: no [cls_label] found in filename")
                        continue
                    
                    cls_label = np.array([int(x) for x in term_split[1]])
                    if len(cls_label) == len(self.CLASSES) - 1:  # Exclude background
                        self.img_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                        self.cls_labels.append(cls_label)
                    else:
                        logger.warning(f"Skipping {img_path}: invalid cls_label length {len(cls_label)}")
                else:
                    logger.warning(f"Mask not found for {img_name} at {mask_path}")
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue

        logger.info(f"Loaded {len(self.img_paths)} valid WSSS samples")
        