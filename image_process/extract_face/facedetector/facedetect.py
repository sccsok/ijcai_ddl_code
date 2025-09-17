import torch
import numpy as np
from .face_utils import norm_crop, norm_crop_mask


def crop_face_frame(img, face_detector, image_size):
    img = np.array(img)
    _img = np.copy(img)

    with torch.no_grad():
        _, landms = face_detector.detect(_img) # shape (1,10)

    if landms == None:
        return None, None, None
    
    landmarks = landms.cpu().detach().numpy().reshape(5, 2).astype(np.int32) # shape (5, 2)
    cropped_img, cropped_msk, loc = norm_crop_mask(img, landmarks, image_size=image_size) # shape (h, w, c)

    return cropped_img, cropped_msk, loc # np.uint8