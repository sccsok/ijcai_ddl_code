import random
import math
import dlib
import sys
from copy import deepcopy
import cv2
import numpy as np
from albumentations import DualTransform, ImageOnlyTransform
from albumentations.augmentations.crops.functional import crop
from scipy.ndimage import binary_erosion, binary_dilation
from skimage import measure, draw
from skimage.transform import PiecewiseAffineTransform, warp
from imutils import face_utils


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")
    
    
class MaskAugmentation(ImageOnlyTransform):
    def __init__(self, config, always_apply=False, p=1):
        super(MaskAugmentation, self).__init__(always_apply, p)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(config['lmd_pth'])
        self.index = ['remove_eyes', 'remove_nose', 'remove_mouth']

    def dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_five_key(self, img):
        # get the five key points by using the landmarks
        faces = self.detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
        landmarks_68 = face_utils.shape_to_np(self.predictor(img, faces[0]))
        leye_center = (landmarks_68[36] + landmarks_68[39]) * 0.5
        reye_center = (landmarks_68[42] + landmarks_68[45]) * 0.5
        nose = landmarks_68[33]
        lmouth = landmarks_68[48]
        rmouth = landmarks_68[54]
        leye_left = landmarks_68[36]
        leye_right = landmarks_68[39]
        reye_left = landmarks_68[42]
        reye_right = landmarks_68[45]
        out = [tuple(x.astype('int32')) for x in [
            leye_center, reye_center, nose, lmouth, rmouth, leye_left, leye_right, reye_left, reye_right
        ]]
        
        return out

    def remove_eyes(self, image, landmarks, opt):
        ##l: left eye; r: right eye, b: both eye
        if opt == 'l':
            (x1, y1), (x2, y2) = landmarks[5:7]
        elif opt == 'r':
            (x1, y1), (x2, y2) = landmarks[7:9]
        elif opt == 'b':
            (x1, y1), (x2, y2) = landmarks[:2]
        else:
            print('wrong region')
        mask = np.zeros_like(image[..., 0])
        line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
        w = self.dist((x1, y1), (x2, y2))
        dilation = int(w // 4)
        if opt != 'b':
            dilation *= 4
        line = binary_dilation(line, iterations=dilation)
        return line

    def remove_nose(self, image, landmarks):
        (x1, y1), (x2, y2) = landmarks[:2]
        x3, y3 = landmarks[2]
        mask = np.zeros_like(image[..., 0])
        x4 = int((x1 + x2) / 2)
        y4 = int((y1 + y2) / 2)
        line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
        w = self.dist((x1, y1), (x2, y2))
        dilation = int(w // 4)
        line = binary_dilation(line, iterations=dilation)
        return line

    def remove_mouth(self, image, landmarks):
        (x1, y1), (x2, y2) = landmarks[3:5]
        mask = np.zeros_like(image[..., 0])
        line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
        w = self.dist((x1, y1), (x2, y2))
        dilation = int(w // 3)
        line = binary_dilation(line, iterations=dilation)
        return line

    def blackout_convex_hull(self, img):
        img_cpy = deepcopy(img)
        try:
            rect = self.detector(img_cpy)[0]
            sp = self.predictor(img_cpy, rect)
            landmarks = np.array([[p.x, p.y] for p in sp.parts()])
            outline = landmarks[[*range(17), *range(26, 16, -1)]]
            Y, X = draw.polygon(outline[:, 1], outline[:, 0])
            cropped_img = np.zeros(img_cpy.shape[:2], dtype=np.uint8)
            cropped_img[Y, X] = 1
            
            y, x = measure.centroid(cropped_img)
            y = int(y)
            x = int(x)
            first = random.random() > 0.5
            if random.random() > 0.5:
                if first:
                    cropped_img[:y, :] = 0
                else:
                    cropped_img[y:, :] = 0
            else:
                if first:
                    cropped_img[:, :x] = 0
                else:
                    cropped_img[:, x:] = 0

            img_cpy[cropped_img > 0] = 0

            return img_cpy
        except Exception as e:
            return img

    def random_deform(self, mask, nrows, ncols, mean=0, std=10):
        h, w = mask.shape[:2]
        rows = np.linspace(0, h - 1, nrows).astype(np.int32)
        cols = np.linspace(0, w - 1, ncols).astype(np.int32)
        rows += np.random.normal(mean, std, size=rows.shape).astype(np.int32)
        rows += np.random.normal(mean, std, size=cols.shape).astype(np.int32)
        rows, cols = np.meshgrid(rows, cols)
        anchors = np.vstack([rows.flat, cols.flat]).T
        assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
        deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
        np.clip(deformed[:, 0], 0, h - 1, deformed[:, 0])
        np.clip(deformed[:, 1], 0, w - 1, deformed[:, 1])

        trans = PiecewiseAffineTransform()
        trans.estimate(anchors, deformed.astype(np.int32))
        warped = warp(mask, trans)
        warped *= mask
        blured = cv2.GaussianBlur(warped, (5, 5), 3)

        return blured

    def remove_landmark(self, image, key_points, i=None):
        if i is None:
            i = random.randint(0, 2)
        if i == 0:
            opt = ['r', 'l', 'b']
            ii = random.randint(0, 2)
            line = self.remove_eyes(image, key_points, opt[ii])
        elif i == 1:
            line = self.remove_mouth(image, key_points)
        else:
            line = self.remove_nose(image, key_points)

        mask = line.astype(np.uint8)
        mask = 1.0 - mask.reshape((mask.shape[0], mask.shape[1], 1))
        image = np.clip(mask * image, 0, 255)

        return image.astype(np.uint8)

    def apply(self, image, **params):
        image_copy = image
        try:
            p = random.uniform(0, 1)
            key_points = self.get_five_key(image)

            if p <= 0.125:
                return self.blackout_convex_hull(image)
            elif 0.125 < p <= 0.25:
                return self.remove_landmark(image, key_points)
            elif 0.25 < p <= 0.375:
                i = random.sample(range(0, 3), 2)
                image = self.remove_landmark(image, key_points, i[0])
                return self.remove_landmark(image, key_points, i[1])
            elif 0.375 < p <= 0.5:
                for i in range(3):
                    image = self.remove_landmark(image, key_points, i)
                return image
        except:
            image = image_copy
            
        return image

    def get_transform_init_args_names(self):
        return ("lmd_pth",)


class Resize4xAndBack(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(Resize4xAndBack, self).__init__(always_apply, p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        scale = random.choice([2, 4])
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (w, h),
                         interpolation=random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST]))
        return img


class RandomSizedCropNonEmptyMaskIfExists(DualTransform):

    def __init__(self, min_max_height, w2h_ratio=[0.7, 1.3], always_apply=False, p=0.5):
        super(RandomSizedCropNonEmptyMaskIfExists, self).__init__(always_apply, p)

        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        cropped = crop(img, x_min, y_min, x_max, y_max)
        return cropped

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]
        mask_height, mask_width = mask.shape[:2]
        crop_height = int(mask_height * random.uniform(self.min_max_height[0], self.min_max_height[1]))
        w2h_ratio = random.uniform(*self.w2h_ratio)
        crop_width = min(int(crop_height * w2h_ratio), mask_width - 1)
        if mask.sum() == 0:
            x_min = random.randint(0, mask_width - crop_width + 1)
            y_min = random.randint(0, mask_height - crop_height + 1)
        else:
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            non_zero_yx = np.argwhere(mask)
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, crop_width - 1)
            y_min = y - random.randint(0, crop_height - 1)
            x_min = np.clip(x_min, 0, mask_width - crop_width)
            y_min = np.clip(y_min, 0, mask_height - crop_height)

        x_max = x_min + crop_height
        y_max = y_min + crop_width
        y_max = min(mask_height, y_max)
        x_max = min(mask_width, x_max)
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def get_transform_init_args_names(self):
        return "min_max_height", "height", "width", "w2h_ratio"


class RandomCopyMove(DualTransform):
    def __init__(self,
                 max_h=0.8,
                 max_w=0.8,
                 min_h=0.05,
                 min_w=0.05,
                 mask_value=255,
                 p=0.5):
        """Apply cope-move manipulation to the image, and change the respective region on the mask to <mask_value>

        Args:
            max_h (float, optional): (0~1), max window height rate to the full height of image . Defaults to 0.5.
            max_w (float, optional): (0~1), max window width rate to the full width of image . Defaults to 0.5.
            min_h (float, optional): (0~1), min window height rate to the full height of image . Defaults to 0.05.
            min_w (float, optional): (0~1), min window width rate to the full width of image . Defaults to 0.05.
            mask_value (int, optional): the value apply the tampered region on the mask. Defaults to 255.
            always_apply (bool, optional): _description_. Defaults to False.
            p (float, optional): _description_. Defaults to 0.5.
        """
        super(RandomCopyMove, self).__init__(p)
        self.max_h = max_h
        self.max_w = max_w
        self.min_h = min_h
        self.min_w = min_w
        self.mask_value = mask_value
        
    def _get_random_window(
        self, 
        img_height, 
        img_width, 
        window_height = None, 
        window_width = None
    ):
        assert self.max_h < 1 and self.max_h > 0 
        assert self.max_w < 1 and self.max_w > 0
        assert self.min_w < 1 and self.min_w > 0
        assert self.min_h < 1 and self.min_h > 0
        
        l_min_h = int(img_height * self.min_h)
        l_min_w = int(img_width * self.min_w)
        l_max_h = int(img_height * self.max_h)
        l_max_w = int(img_width * self.max_w)
        
        if window_width == None or window_height == None:
            window_h = np.random.randint(l_min_h, l_max_h)
            window_w = np.random.randint(l_min_w, l_max_w)
        else:
            window_h = window_height
            window_w = window_width

        # position of left up corner of the window
        pos_h = np.random.randint(0, img_height - window_h)
        pos_w = np.random.randint(0, img_width - window_w)
        
        return pos_h, pos_w , window_h, window_w
        
        
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        image = img.copy()
        H, W, _ = image.shape
        # copy region:
        c_pos_h, c_pos_w, c_window_h, c_window_w = self._get_random_window(H, W)
        
        # past region, window size is defined by copy region:
        self.p_pos_h, self.p_pos_w, self.p_window_h, self.p_window_w = self._get_random_window(H, W, c_window_h, c_window_w)
        
        copy_region = image[
            c_pos_h: c_pos_h + c_window_h, 
            c_pos_w: c_pos_w + c_window_w, 
            : 
        ]
        image[
            self.p_pos_h : self.p_pos_h + self.p_window_h,
            self.p_pos_w : self.p_pos_w + self.p_window_w,
            :
        ] = copy_region
        return image
        

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """
        change the mask of manipulated region to 1
        """
    
        manipulated_region =  np.full((self.p_window_h, self.p_window_w), 1)
        img = img.copy()
        img[
            self.p_pos_h : self.p_pos_h + self.p_window_h,
            self.p_pos_w : self.p_pos_w + self.p_window_w,
        ] = self.mask_value
        return img
    # must be implemented for string output when print(RandomCopyMove())
    def get_transform_init_args_names(self):
        return ("max_h", "max_w", "min_h", "min_w", "mask_value", "always_apply", "p")
        
class RandomInpainting(DualTransform):
    def __init__(self,
        max_h = 0.8,
        max_w = 0.8,
        min_h = 0.05,
        min_w = 0.05,
        mask_value = 255,
        p = 0.5,  
    ):
        super(RandomInpainting, self).__init__(p)
        self.max_h = max_h
        self.max_w = max_w
        self.min_h = min_h
        self.min_w = min_w
        self.mask_value = mask_value
    def _get_random_window(
        self, 
        img_height, 
        img_width, 
    ):
        assert self.max_h < 1 and self.max_h > 0 
        assert self.max_w < 1 and self.max_w > 0
        assert self.min_w < 1 and self.min_w > 0
        assert self.min_h < 1 and self.min_h > 0
        
        l_min_h = int(img_height * self.min_h)
        l_min_w = int(img_width * self.min_w)
        l_max_h = int(img_height * self.max_h)
        l_max_w = int(img_width * self.max_w)
    
        window_h = np.random.randint(l_min_h, l_max_h)
        window_w = np.random.randint(l_min_w, l_max_w)

        # position of left up corner of the window
        pos_h = np.random.randint(0, img_height - window_h)
        pos_w = np.random.randint(0, img_width - window_w)
        
        return pos_h, pos_w , window_h, window_w
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.copy()
        img = np.uint8(img)
        H, W, C = img.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        # inpainting region
        self.pos_h, self.pos_w , self.window_h, self.window_w = self._get_random_window(H, W)
        mask[
            self.pos_h : self.pos_h+ self.window_h,
            self.pos_w : self.pos_w + self.window_w,
        ] = 1
        inpaint_flag = cv2.INPAINT_TELEA if random.random() > 0.5 else cv2.INPAINT_NS
        img = cv2.inpaint(img, mask, 3,inpaint_flag)
        return img
    
    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """
        change the mask of manipulated region to 1
        """
        img = img.copy()
        img[
            self.pos_h : self.pos_h+ self.window_h,
            self.pos_w : self.pos_w + self.window_w,
        ] = self.mask_value
        return img
    
    # must be implemented for string output when print(RandomInpainting())
    def get_transform_init_args_names(self):
        return ("max_h", "max_w", "min_h", "min_w", "mask_value", "always_apply", "p")