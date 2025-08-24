from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
from face_rotation import AlignAndCropFace
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def get_data_loaders(train_dir, test_dir, image_size=(320,320), batch_size=32, num_workers=0, pin_memory=True):
    I = InterpolationMode
    w, h = image_size

    align = AlignAndCropFace(out_size=image_size, enlarge=1.3, fill=(128,128,128))

    # 단일 보간으로 320 고정
    to_320 = transforms.Resize((h, w), interpolation=I.BICUBIC)

    # Train: 보수적 증강(강회전/퍼스펙티브 제거)
    train_transform = transforms.Compose([
        align,
        to_320,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ], p=0.5),
        # 선택: 아주 약한 affine만 쓸 때 (원하면 주석 해제)
        # transforms.RandomAffine(degrees=5, translate=(0.02,0.02), scale=(0.95,1.05), interpolation=I.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Test: 최소 변환
    test_transform = transforms.Compose([
        align,
        to_320,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=test_transform)

    def robust_collate(batch):
        batch = [(x,y) for (x,y) in batch if x is not None]
        if len(batch)==0:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        imgs, ys = [], []
        for img,y in batch:
            if not torch.is_tensor(img):
                if isinstance(img, Image.Image):
                    img = transforms.ToTensor()(img)
                else:
                    continue
            if img.dim()!=3:
                continue
            C,H,W = img.shape
            if (H,W)!=(h,w):
                img = F.interpolate(img.unsqueeze(0), size=(h,w), mode="bilinear", align_corners=False).squeeze(0)
            imgs.append(img); ys.append(y)
        if len(imgs)==0:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        return torch.stack(imgs,0), torch.tensor(ys, dtype=torch.long)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=(num_workers>0), collate_fn=robust_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             persistent_workers=(num_workers>0), collate_fn=robust_collate)
    return train_loader, test_loader
