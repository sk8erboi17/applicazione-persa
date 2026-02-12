import albumentations as alb
from albumentations.pytorch import ToTensorV2

train_transform = alb.Compose(
    [
        alb.Affine(scale=(0.85, 1.0), rotate=(-1, 1), border_mode=0, interpolation=3, p=0.15),
        alb.RGBShift(r_shift_limit=15, g_shift_limit=15,
                     b_shift_limit=15, p=0.3),
        alb.GaussNoise(std_range=(0.03, 0.07), p=.2),
        alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
        alb.ImageCompression(quality_range=(75, 95), compression_type="jpeg", p=.3),
        alb.ToGray(p=1.0),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        ToTensorV2(),
    ]
)
test_transform = alb.Compose(
    [
        alb.ToGray(p=1.0),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        ToTensorV2(),
    ]
)
