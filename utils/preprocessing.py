from PIL import Image
from torchvision import transforms

def preprocessing(path, ratio, resize_height, resize_width):
    """
    Loads an image, resizes it for HR and LR, and normalizes pixel values.
    """
    img_hr = Image.open(path).convert("RGB")
    to_tensor = transforms.ToTensor()

    transform_hr = transforms.Compose([
        transforms.Resize((resize_height, resize_width), interpolation=transforms.InterpolationMode.BICUBIC),
        to_tensor
    ])
    hr = transform_hr(img_hr)

    h, w = hr.shape[1], hr.shape[2]
    lr_height, lr_width = h // ratio, w // ratio

    transform_lr = transforms.Compose([
        transforms.Resize((lr_height, lr_width), interpolation=transforms.InterpolationMode.BICUBIC),
        to_tensor
    ])
    lr = transform_lr(img_hr)

    return lr, hr