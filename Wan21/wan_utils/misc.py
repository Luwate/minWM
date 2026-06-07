import numpy as np
import random
import torch


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict


def crop_batch(batch, image_or_video_shape):
    # image_or_video_shape: [B, F, C, H, W]
    if "clean_latent" in batch:
        latent = batch["clean_latent"]
        # latent shape: [B, F, C, H, W]
        _, F_cfg, _, H_cfg, W_cfg = image_or_video_shape
        _, F_lat, C_lat, H_lat, W_lat = latent.shape
        if H_lat > H_cfg or W_lat > W_cfg:
            h_start = (H_lat - H_cfg) // 2
            w_start = (W_lat - W_cfg) // 2
            batch["clean_latent"] = latent[:, :, :, h_start:h_start+H_cfg, w_start:w_start+W_cfg]
        if F_lat > F_cfg:
            batch["clean_latent"] = batch["clean_latent"][:, :F_cfg, ...]
            if "viewmats" in batch:
                batch["viewmats"] = batch["viewmats"][:, :F_cfg, ...]
            if "Ks" in batch:
                batch["Ks"] = batch["Ks"][:, :F_cfg, ...]

    if "ode_latent" in batch:
        latent = batch["ode_latent"]
        # latent shape: [B, S, F, C, H, W]
        _, F_cfg, _, H_cfg, W_cfg = image_or_video_shape
        _, S_lat, F_lat, C_lat, H_lat, W_lat = latent.shape
        if H_lat > H_cfg or W_lat > W_cfg:
            h_start = (H_lat - H_cfg) // 2
            w_start = (W_lat - W_cfg) // 2
            batch["ode_latent"] = latent[:, :, :, :, h_start:h_start+H_cfg, w_start:w_start+W_cfg]
        if F_lat > F_cfg:
            batch["ode_latent"] = batch["ode_latent"][:, :, :F_cfg, ...]
            if "viewmats" in batch:
                batch["viewmats"] = batch["viewmats"][:, :F_cfg, ...]
            if "Ks" in batch:
                batch["Ks"] = batch["Ks"][:, :F_cfg, ...]
    return batch

