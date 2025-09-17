import os
import cv2
import dlib
import numpy as np
from PIL import Image, ImageDraw
import random

def detect_and_crop_face(img_path, detector, predictor, margin=10):
    """
    Detect the largest face in an image and crop it based on facial landmarks convex hull.
    Returns:
        - face_rgba: Cropped RGBA face image with alpha mask (hull).
        - bbox: (l, t, r, b) of cropped region in original image.
        - hull_pts: Convex hull points (relative to crop top-left).
    Returns None if no face is detected.
    """
    img = dlib.load_rgb_image(img_path)
    dets = detector(img, 1)
    if not dets:
        return None

    # Select the largest face
    rect = max(dets, key=lambda r: r.width() * r.height())
    l, t, r_, b = rect.left(), rect.top(), rect.right(), rect.bottom()
    h, w = img.shape[:2]

    # Apply margin
    l = max(0, l - margin)
    t = max(0, t - margin)
    r_ = min(w, r_ + margin)
    b = min(h, b + margin)
    crop = img[t:b, l:r_]

    # Get facial landmarks and compute convex hull
    shape = predictor(img, rect)
    pts = np.array([[p.x - l, p.y - t] for p in shape.parts()])
    hull = cv2.convexHull(pts)

    # Generate binary mask from convex hull
    mask_full = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask_full, hull, 255)
    ys, xs = np.where(mask_full > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # Final crop and mask
    face_crop = crop[y0:y1+1, x0:x1+1]
    mask_crop = mask_full[y0:y1+1, x0:x1+1]

    # Combine RGB with alpha channel
    face_rgba = cv2.cvtColor(face_crop, cv2.COLOR_RGB2RGBA)
    face_rgba[:, :, 3] = mask_crop

    return face_rgba, (l + x0, t + y0, l + x1, t + y1), hull - [x0, y0]

def paste_random_faces(sources, target_img, detector, predictor, output_dir,
                       min_faces=1, max_faces=3):
    """
    Paste 1 to 3 (random) detected convex hull faces from sources into the target image.
    - Randomly select faces from sources.
    - Resize based on target face width.
    - Paste with alpha blending.
    Returns True on success, False if no target or source face is detected.
    """
    dets2 = detector(target_img, 1)
    if not dets2:
        return False
    rect2 = max(dets2, key=lambda r: r.width() * r.height())
    dx1, dy1, dx2, dy2 = rect2.left(), rect2.top(), rect2.right(), rect2.bottom()
    dst_w = dx2 - dx1

    pool = sources.copy()
    random.shuffle(pool)
    crops = []
    tries = 0

    # Try detecting up to max_faces
    while tries < max_faces and pool:
        item = detect_and_crop_face(pool.pop(), detector, predictor)
        if item:
            crops.append(item)
        tries += 1

    # Ensure at least min_faces
    while len(crops) < min_faces and pool:
        item = detect_and_crop_face(pool.pop(), detector, predictor)
        if item:
            crops.append(item)

    if not crops:
        return False

    face_count = random.randint(1, len(crops))
    pil_bg = Image.fromarray(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    mask_canvas = Image.new('L', pil_bg.size, 0)
    draw = ImageDraw.Draw(mask_canvas)
    W, H = pil_bg.size

    for face_rgba, _, _ in random.sample(crops, face_count):
        h_s, w_s = face_rgba.shape[:2]

        # Optional: Randomly crop half-face (left/right/top/bottom)
        if random.random() < 0.5:
            if random.random() < 0.5:
                face_rgba = face_rgba[:, :w_s // 2, :] if random.random() < 0.5 else face_rgba[:, w_s // 2:, :]
            else:
                face_rgba = face_rgba[:h_s // 2, :, :] if random.random() < 0.5 else face_rgba[h_s // 2:, :, :]
            h_s, w_s = face_rgba.shape[:2]

        # Resize to ~0.4–0.6 of target face width
        scale = (random.random() * 0.2 + 0.4) * dst_w / w_s
        nw, nh = int(w_s * scale), int(h_s * scale)
        face_pil = Image.fromarray(face_rgba).resize((nw, nh), Image.LANCZOS)

        # Random paste position
        x = random.randint(0, max(0, W - nw))
        y = random.randint(0, max(0, H - nh))

        # Paste onto image and update mask
        pil_bg.paste(face_pil, (x, y), face_pil)
        draw.bitmap((x, y), face_pil.split()[3], fill=255)

    # Save output images and masks
    os.makedirs(os.path.join(output_dir, 'fake'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'mask'), exist_ok=True)
    base = os.path.splitext(os.path.basename(target_img_path))[0]
    out_img = os.path.join(output_dir, 'fake', f"{base}.png")
    out_mask = os.path.join(output_dir, 'mask', f"{base}.png")
    pil_bg.save(out_img)
    mask_canvas.save(out_mask)
    return True

if __name__ == '__main__':
    src_dir = '../../face/train/real'
    output_dir = '../../ddl_data/self/ps_random'
    num_sources = 30000
    num_targets = 10000

    # Initialize face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_81_face_landmarks.dat')

    # Load and sample image paths
    all_png = [os.path.join(src_dir, f) for f in os.listdir(src_dir)
               if f.lower().endswith('.png')]
    sampled = random.sample(all_png, num_sources + num_targets)
    sources = sampled[:num_sources]
    targets = sampled[num_sources:]

    # Run synthesis
    success = 0
    for target_img_path in targets:
        if success >= num_targets:
            break
        target_img = cv2.imread(target_img_path)
        if target_img is None:
            continue
        ok = paste_random_faces(
            sources, target_img, detector, predictor, output_dir,
            min_faces=1, max_faces=3
        )
        if ok:
            success += 1
            print(f"[{success}/{num_targets}] Success: {os.path.basename(target_img_path)}")
        else:
            print(f"[Skipped] No face detected: {os.path.basename(target_img_path)}")
