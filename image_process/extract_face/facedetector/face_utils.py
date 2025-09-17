# 2021.12.22 richard
import cv2
import numpy as np

import torch
from skimage.transform import SimilarityTransform

from .data import cfg_re50
from .layers.functions.prior_box import PriorBox
from .utils.box_utils import decode, decode_landm
from .models.retinaface import RetinaFace
from PIL import Image

# mean and std of deepfake detection model
# MEAN, STD = [0.4479, 0.3744, 0.3473], [0.2537, 0.2502, 0.2424]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# mean and std of Retinaface detection model
RetinaMEAN, RetinaSTD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def norm_crop(img, landmark, image_size=112):
    ARCFACE_SRC = np.array([[
        [122.5, 141.25],
        [197.5, 141.25],
        [160.0, 178.75],
        [137.5, 225.25],
        [182.5, 225.25]
    ]], dtype=np.float32)

    def estimate_norm(lmk):
        assert lmk.shape == (5, 2)

        tform = SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = np.inf
        src = ARCFACE_SRC

        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]

        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))

        if error < min_error:
            min_error = error
            min_M = M
            min_index = i

        return min_M, min_index

    # [2021.11.11] ATTENTION: without estimating the transformation matrix, 
    # face region will not be properly cropped out
    M, _ = estimate_norm(landmark)
    # M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    # img shape: h, w, c
    try:
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    except:
        print(M)
        raise cv2.error
    # code below will save raw(img)/cropped(warped) image   
    # std = np.repeat(np.array(STD).reshape(1,1,3), warped.shape[0], 0)
    # std = np.repeat(std, warped.shape[1], 1)

    # mean = np.repeat(np.array(MEAN).reshape(1,1,3), warped.shape[0], 0)
    # mean = np.repeat(mean, warped.shape[1], 1)

    # warped = warped*std + mean
    # warped *= 255
    # Image.fromarray(warped.astype(np.uint8)).save("warped.png")

    return warped


# no alignment, adaptive scaling face with no resize operation
# -*- coding: utf-8 -*-
# @Date     : 2025/01/21
# @Author   : Chao Shuai
# @File     : face_utils.py
def norm_crop_mask(img, landmark, image_size=112):
    image_width, image_height, _ = img.shape
    x_min = min(landmark[:, 0])
    x_max = max(landmark[:, 0])
    y_min = min(landmark[:, 1])
    y_max = max(landmark[:, 1])
    
    # padding_ratio = [0.75, 0.8]
    # a larger factor for localization
    padding_ratio = [1.5, 1.5]
    width = x_max - x_min
    height = y_max - y_min
    padding_ratio[0] = padding_ratio[0] * (height / width)
    padding_ratio[1] = padding_ratio[1] * (width / height)
    dx = int(width * padding_ratio[0])
    dy = int(height * padding_ratio[1])
    
    y_min, y_max, x_min, x_max =  max(0, y_min - dy), min(image_width, y_max + dy), max(0, x_min - dx), min(image_height, x_max + dx)
    warped = img[y_min :y_max, x_min :x_max]
    
    mask = np.zeros(img.shape[:2], dtype=np.uint8) 
    mask[y_min :y_max, x_min :x_max] = 255

    return warped, mask, [y_min, y_max, x_min, x_max]


def batch_decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [bs, num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    bs = loc.shape[0]
    _priors = priors.unsqueeze(0).repeat(bs,1,1)
    boxes = torch.cat((
        _priors[:, :, :2] + loc[:, :, :2] * variances[0] * _priors[:, :, 2:],
        _priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1])), 2)
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


def batch_decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [bs, num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """

    bs = pre.shape[0]
    _priors = priors.unsqueeze(0).repeat(bs,1,1)
    landms = torch.cat((_priors[:, :, :2] + pre[:, :, :2] * variances[0] * _priors[:, :, 2:],
                        _priors[:, :, :2] + pre[:, :, 2:4] * variances[0] * _priors[:, :, 2:],
                        _priors[:, :, :2] + pre[:, :, 4:6] * variances[0] * _priors[:, :, 2:],
                        _priors[:, :, :2] + pre[:, :, 6:8] * variances[0] * _priors[:, :, 2:],
                        _priors[:, :, :2] + pre[:, :, 8:10] * variances[0] * _priors[:, :, 2:],
                        ), dim=2)
    return landms


class FaceDetector:
    def __init__(self, device="cuda", _reinit_cuda = False, _max_inds=True, confidence_threshold=0.9):
        self.device = device
        self.confidence_threshold = confidence_threshold

        self.cfg = cfg = cfg_re50
        self.variance = cfg["variance"]
        cfg["pretrain"] = False
        self.net = RetinaFace(cfg=self.cfg, phase="test").to(self.device).eval()
        self.decode_param_cache = {}
        self._reinit_cuda = _reinit_cuda    # recast model to current device (specified by torch.cuda.set_device())
        # this is especially useful in multi-worker dataloader with face detection
        # 
        # Usage:
        # def Customdataset(Dataset):
        #     ....
        #     def __getitem__(self,idx):
        #         ...
        #         img = facedetector.detect(img)
        # def w_init_fn(id):
        #     torch.cuda.set_device(cuda_list[id%len(cuda_list)])
        # ...
        # dataloader(dataset, num_workers=1, worker_init_fn=w_init_fn, mulprocessing_context="spawn")
        # 

        self._max_inds = _max_inds          # return landms and boxes with max confidence

    def load_checkpoint(self, path):
        self.net.load_state_dict(torch.load(path))

    def decode_params(self, height, width):
        cache_key = (height, width)

        try:
            return self.decode_param_cache[cache_key]
        except KeyError:
            priorbox = PriorBox(self.cfg, image_size=(height, width))
            priors = priorbox.forward()

            prior_data = priors.data
            scale = torch.Tensor([width, height] * 2)
            scale1 = torch.Tensor([width, height] * 5)

            result = (prior_data, scale, scale1)
            self.decode_param_cache[cache_key] = result

            return result

    """
    detect face in single image
    PARAM: img: PIL or numpy.array (on cpu)
    RETURN: all boxes and landmarks or box and landmark with largest confidence, (controled by self._max_inds)
            (on cpu)
    """
    def py_cpu_nms(self, dets, thresh):
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def detect(self, img):
        prior_data, scale, scale1 = self.decode_params(*img.shape[:2])

        # REF: test_fddb.py
        img = np.float32(img)
        img -= (104, 117, 123)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)

        img = img.to(self.device, dtype=torch.float32)

        loc, conf, landms = self.net(img)

        loc = loc.cpu()
        conf = conf.cpu()
        landms = landms.cpu()

        # Decode results
        boxes = decode(loc.squeeze(0), prior_data, self.variance)
        boxes = boxes * scale
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = decode_landm(landms.squeeze(0), prior_data, self.variance)
        landms = landms * scale1
        
        landms, boxes = landms.cpu().numpy(), boxes.cpu().numpy()
        
        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        
        # keep top-K before NMS
        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.py_cpu_nms(dets, 0.4)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        boxes, scores = dets[:, 0:4], dets[:, 4]
        
        if not self._max_inds:
            return boxes, landms
        
        max_ind = scores.argmax(0)
        boxes = boxes[max_ind]
        landms = landms[max_ind]

        return boxes, landms

    def batch_detect(self, batch):
        """
        NOTICE 2021.12.22,richard: 
        This does not necessarily outperform detect() that can be combined with multiple workers in dataloader

        detect faces from a batch of images
        PARAM: batch: torch.tensor of shape [b, c, h, w]
        RETURN: indexes, boxes and landmarks tensor (on cpu)
        """

        if self._reinit_cuda and torch.cuda.current_device() != -1:
            self.device = f"cuda:{torch.cuda.current_device()}"
            self.net.to(self.device)

        device = self.device
        batch = batch.to(device)

        bs,ch,h,w = batch.shape

        prior_data, scale, scale1 = self.decode_params(h,w)
        prior_data, scale, scale1 = prior_data.to(device), scale.to(device), scale1.to(device)
        # shape: [num_priors, 4], 4 ,10
        
        # mean and std for deepfake detection
        pre_mean = torch.tensor(MEAN).view(1,ch,1,1).repeat(bs, 1, h, w).to(device)
        pre_std = torch.tensor(STD).view(1,ch,1,1).repeat(bs, 1, h, w).to(device)
        
        # NOTICE 2021.12.22 richard
        # Usually, those images in batch have been normalized in __getitem__ of dataset
        # Thus, it is necessary to recover from normalization and then implement new normalization
        # with respect to Retinaface network（minus mean value of RGB)
        # If you use another mean and std in __getitem__, remember to modify the value of global variable MEAN and STD

        batch = (batch * pre_std + pre_mean)* 255
        batch -= RetinaMEAN*255

        loc, conf, landms = self.net(batch)
        # print(f"confidence shape{conf.shape}, loc shape:{loc.shape}")
        # shape [batch, num_priors, 4]

        # Decode results
        boxes = batch_decode(loc, prior_data, self.variance)
        boxes = boxes * scale

        landms = batch_decode_landm(landms, prior_data, self.variance)
        landms = landms * scale1

        inds = conf[:, :, 1] > self.confidence_threshold

        return inds.cpu(), boxes.cpu(), landms.cpu()
