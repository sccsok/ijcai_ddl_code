import os
import random
import itertools
import dlib
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def paste_face_nonrect(
    img1_path: str,
    img2_path: str,
    output_dir: str,
    face_detector: dlib.fhog_object_detector,
    face_predictor: dlib.shape_predictor,
    extra_scale: float = 1.0
):
    """
    Extract the largest face from img1 as a polygon mask (based on facial landmarks convex hull),
    resize it to fit the largest face in img2, and paste it over. Save the composite and its mask.
    """

    # Step 1: Detect largest face in source image
    img1 = dlib.load_rgb_image(img1_path)
    dets1 = face_detector(img1, 1)
    if not dets1:
        print(f"[WARN] No face detected in source: {img1_path}")
        return
    rect1 = max(dets1, key=lambda r: r.width() * r.height())

    # Step 2: Create convex hull mask using facial landmarks
    shape1 = face_predictor(img1, rect1)
    pts1 = np.array([[p.x, p.y] for p in shape1.parts()])
    hull = cv2.convexHull(pts1)
    full_mask = np.zeros(img1.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(full_mask, hull, 255)

    # Step 3: Crop minimal bounding box around the mask
    ys, xs = np.where(full_mask > 0)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    face_rgb_crop = img1[y1:y2+1, x1:x2+1]
    mask_crop = full_mask[y1:y2+1, x1:x2+1]

    # Convert RGB + mask to RGBA
    face_rgba_crop = cv2.cvtColor(face_rgb_crop, cv2.COLOR_RGB2RGBA)
    face_rgba_crop[:, :, 3] = mask_crop

    # Step 4: Detect largest face in target image
    img2_bgr = cv2.imread(img2_path)
    if img2_bgr is None:
        raise FileNotFoundError(f"Cannot read target image: {img2_path}")
    img2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
    dets2 = face_detector(img2, 1)
    if not dets2:
        print(f"[WARN] No face detected in target: {img2_path}")
        return
    rect2 = max(dets2, key=lambda r: r.width() * r.height())
    dx1, dy1, dx2, dy2 = rect2.left(), rect2.top(), rect2.right(), rect2.bottom()
    dst_w, dst_h = dx2 - dx1, dy2 - dy1

    # Step 5: Resize source face crop to match target face region
    h_crop, w_crop = face_rgba_crop.shape[:2]
    scale = max(dst_w / w_crop, dst_h / h_crop) * extra_scale
    new_w, new_h = int(w_crop * scale), int(h_crop * scale)
    resized_face = cv2.resize(face_rgba_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Step 6: Calculate paste coordinates (center aligned)
    cx2, cy2 = (dx1 + dx2) / 2, (dy1 + dy2) / 2
    cx1, cy1 = w_crop / 2, h_crop / 2
    off_x, off_y = cx1 * scale, cy1 * scale
    paste_x = int(cx2 - off_x)
    paste_y = int(cy2 - off_y)

    # Step 7: Paste RGBA face onto target image using PIL and save
    pil_bg = Image.fromarray(img2)
    pil_face = Image.fromarray(resized_face)
    pil_bg.paste(pil_face, (paste_x, paste_y), pil_face)

    os.makedirs(os.path.join(output_dir, 'fake'), exist_ok=True)
    base1 = os.path.splitext(os.path.basename(img1_path))[0]
    base2 = os.path.splitext(os.path.basename(img2_path))[0]
    out_path = os.path.join(output_dir, 'fake', f"{base1}_on_{base2}.png")
    pil_bg.save(out_path)
    print(f"[OK] Saved: {out_path}")

    # Step 8: Create corresponding binary mask (same size as target)
    mask_canvas = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.uint8)
    y0, y1_ = paste_y, paste_y + new_h
    x0, x1_ = paste_x, paste_x + new_w

    # Clip paste region to image bounds
    y0_clip, y1_clip = max(0, y0), min(mask_canvas.shape[0], y1_)
    x0_clip, x1_clip = max(0, x0), min(mask_canvas.shape[1], x1_)

    # Calculate corresponding indices in resized mask
    mask_y0 = y0_clip - y0
    mask_x0 = x0_clip - x0
    mask_y1 = mask_y0 + (y1_clip - y0_clip)
    mask_x1 = mask_x0 + (x1_clip - x0_clip)

    # Paste mask
    mask_canvas[y0_clip:y1_clip, x0_clip:x1_clip] = resized_mask[mask_y0:mask_y1, mask_x0:mask_x1]

    # Save binary mask
    os.makedirs(os.path.join(output_dir, 'mask'), exist_ok=True)
    mask_img = Image.fromarray(mask_canvas)
    mask_path = os.path.join(output_dir, 'mask', f"{base1}_on_{base2}.png")
    mask_img.save(mask_path)
    print(f"[OK] Mask saved: {mask_path}")


# Example usage
if __name__ == '__main__':
    # Load dlib models
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = './shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)

    # Path setup
    src_dir = '../../face/train/real'
    output_dir = '../../ddl_data/self/ps_face'
    target_count = 10000

    # Sample 400 PNG images
    all_png = [f for f in os.listdir(src_dir) if f.lower().endswith('.png')]
    if len(all_png) < 400:
        raise RuntimeError(f"Not enough PNGs in source directory. Found {len(all_png)}.")
    sampled = random.sample(all_png, 400)
    imgs_a = sampled[:200]
    imgs_b = sampled[200:]

    # Generate all A×B combinations and shuffle
    combos = list(itertools.product(imgs_a, imgs_b))
    random.shuffle(combos)

    # Synthesize faces
    success = 0
    pbar = tqdm(total=target_count, desc="Synthesizing")

    for img1, img2 in combos:
        if success >= target_count:
            break

        path1 = os.path.join(src_dir, img1)
        path2 = os.path.join(src_dir, img2)

        try:
            paste_face_nonrect(
                img1_path=path1,
                img2_path=path2,
                output_dir=output_dir,
                face_detector=face_detector,
                face_predictor=face_predictor
            )
            success += 1
            pbar.update(1)
        except Exception:
            continue  # Skip failed cases

    pbar.close()
    if success < target_count:
        print(f"Warning: Only {success} synthesized. Exhausted all combinations.")
    else:
        print(f"Done: {target_count} synthesized.")
