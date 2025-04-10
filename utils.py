# utils.py
import torch
import torchvision.transforms.functional as TF

def get_transforms(resize_to=224):
    """
    Transforms video tensor for VideoMAE:
    - Input: (T, H, W, C) or (T, C, H, W)
    - Output: (T, C, H, W), with each frame resized to 224×224
    - Normalized with ImageNet mean/std
    """

    def transform(video):
        if not isinstance(video, torch.Tensor):
            video = torch.as_tensor(video)
        video = video.float()

        # Convert to [0, 1] if in [0, 255]
        if video.max() > 1.0:
            video = video / 255.0

        # Ensure shape is (T, C, H, W)
        if video.dim() != 4:
            raise ValueError(f"Expected 4D video tensor, got shape: {video.shape}")
        if video.shape[-1] == 3:  # (T, H, W, C)
            video = video.permute(0, 3, 1, 2)
        elif video.shape[1] != 3:
            raise ValueError(f"Expected 3 channels in dim 1, got shape: {video.shape}")

        # Resize each frame to (resize_to, resize_to)
        video = torch.stack([TF.resize(frame, [resize_to, resize_to], antialias=True) for frame in video])

        # Normalize with ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=video.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=video.device).view(1, 3, 1, 1)
        video = (video - mean) / std

        return video  # shape: (T, C, H, W)

    return transform
