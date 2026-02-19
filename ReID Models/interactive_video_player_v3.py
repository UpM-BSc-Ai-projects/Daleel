import cv2
import sys
import os
import csv
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────────────
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.6
FONT_THICKNESS  = 2
TEXT_COLOR      = (0, 0, 255)       # red
SHADOW_COLOR    = (0, 0, 0)         # black drop-shadow for readability
BOX_COLOR       = (0, 255, 0)       # green bounding box
BOX_THICKNESS   = 2
PADDING         = 8                 # pixels from edge
# ─────────────────────────────────────────────────────────────────────────────

mouse_x, mouse_y = -1, -1
mouse_inside     = False


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_inside
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
        mouse_inside = True
    elif event == cv2.EVENT_MOUSELEAVE:
        mouse_inside = False


def draw_text_with_shadow(frame, text, pos, font_scale=None, thickness=None, color=TEXT_COLOR):
    """Draw text with a 1-pixel black shadow for contrast on any background."""
    fs = font_scale if font_scale is not None else FONT_SCALE
    th = thickness  if thickness  is not None else FONT_THICKNESS
    sx, sy = pos[0] + 1, pos[1] + 1
    cv2.putText(frame, text, (sx, sy), FONT, fs, SHADOW_COLOR, th, cv2.LINE_AA)
    cv2.putText(frame, text, pos,       FONT, fs, color,   th, cv2.LINE_AA)


def load_csv(csv_path):
    """
    Load detection CSV into a dict keyed by frame number.
    Stores raw normalised YOLO values (xc, yc, bw, bh) — converted to pixels
    at draw time so we don't need frame dimensions up front.

    Returns: { frame_no (int): [ {person_id, xc, yc, bw, bh}, ... ] }
    """
    detections = defaultdict(list)
    if csv_path is None or not os.path.exists(csv_path):
        return detections

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_no  = int(row["frame_no"])
                person_id = row["person_id"]
                xc  = float(row["x_center"])
                yc  = float(row["y_center"])
                bw  = float(row["width"])
                bh  = float(row["height"])
                detections[frame_no].append({
                    "person_id": person_id,
                    "xc": xc, "yc": yc,
                    "bw": bw, "bh": bh,
                })
            except (KeyError, ValueError):
                continue  # skip malformed rows

    print(f"  Loaded {sum(len(v) for v in detections.values())} detections "
          f"across {len(detections)} frames from CSV.")
    return detections


def find_csv(video_path, csv_dir):
    """
    Search csv_dir for a CSV file whose stem matches the video filename stem.
    E.g. 'footage/cam1.mp4'  →  '<csv_dir>/cam1.csv'
    Falls back to a case-insensitive match.
    """
    if csv_dir is None:
        return None

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    candidate  = os.path.join(csv_dir, video_stem + ".csv")

    if os.path.exists(candidate):
        print(f"  Found matching CSV: {candidate}")
        return candidate

    # Case-insensitive fallback
    try:
        for fname in os.listdir(csv_dir):
            if fname.lower() == (video_stem + ".csv").lower():
                full = os.path.join(csv_dir, fname)
                print(f"  Found matching CSV (case-insensitive): {full}")
                return full
    except OSError:
        pass

    print(f"  No matching CSV found for '{video_stem}' in '{csv_dir}'. "
          "Running without bounding boxes.")
    return None


def yolo_to_pixel(xc, yc, bw, bh, img_w, img_h):
    """Convert normalised YOLO xywh → pixel (x1, y1, x2, y2)."""
    cx     = xc * img_w
    cy     = yc * img_h
    half_w = (bw * img_w) / 2
    half_h = (bh * img_h) / 2
    return (int(cx - half_w), int(cy - half_h),
            int(cx + half_w), int(cy + half_h))


def get_hovered_detection(mx, my, frame_no, detections, img_w, img_h):
    """
    Return (detection_dict, (x1,y1,x2,y2)) for the box under the cursor on
    the given frame, or (None, None) if no hit.
    When multiple boxes overlap, the smallest area (most specific) wins.
    """
    hits = []
    for det in detections.get(frame_no, []):
        x1, y1, x2, y2 = yolo_to_pixel(
            det["xc"], det["yc"], det["bw"], det["bh"], img_w, img_h
        )
        if x1 <= mx <= x2 and y1 <= my <= y2:
            area = (x2 - x1) * (y2 - y1)
            hits.append((area, det, x1, y1, x2, y2))

    if not hits:
        return None, None

    hits.sort(key=lambda t: t[0])  # smallest area = best hit
    _, det, x1, y1, x2, y2 = hits[0]
    return det, (x1, y1, x2, y2)


def play_video(video_path, csv_dir=None):
    global mouse_inside

    if not os.path.exists(video_path):
        print(f"Error: file not found → {video_path}")
        sys.exit(1)

    # ── Load detections ──────────────────────────────────────────────────────
    csv_path   = find_csv(video_path, csv_dir)
    detections = load_csv(csv_path)

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: could not open video.")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    delay        = max(1, int(1000 / fps))

    window_name = "Interactive Video Player  |  Q/ESC = quit  |  SPACE = pause  |  Left/Right = step frames"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    paused    = False
    frame_idx = 0
    frame     = None

    print(f"\nPlaying : {video_path}")
    print(f"  {total_frames} frames  |  {fps:.2f} fps")
    print("Controls: SPACE = pause/resume   ← → = step one frame   Q / ESC = quit\n")

    while True:
        # ── Advance frame ────────────────────────────────────────────────────
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame is None:
            continue

        display          = frame.copy()
        img_h, img_w     = display.shape[:2]
        cursor_in_bounds = mouse_inside and 0 <= mouse_x < img_w and 0 <= mouse_y < img_h

        # ── Hover: find & draw bounding box ──────────────────────────────────
        if cursor_in_bounds and detections:
            hovered_det, bbox = get_hovered_detection(
                mouse_x, mouse_y, frame_idx, detections, img_w, img_h
            )
            if hovered_det is not None:
                x1, y1, x2, y2 = bbox

                # Bounding box rectangle
                cv2.rectangle(display, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

                # Person ID label — placed just below the box
                label = f"ID: {hovered_det['person_id']}"
                (lw, lh), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
                label_x = max(0, min(x1, img_w - lw - 2))
                label_y = y2 + lh + PADDING // 2
                # If label would go off the bottom, put it above the box instead
                if label_y > img_h - PADDING:
                    label_y = max(lh + 2, y1 - PADDING // 2)
                draw_text_with_shadow(display, label, (label_x, label_y), color=BOX_COLOR)

        # ── Frame counter — top-right ─────────────────────────────────────────
        frame_text  = f"Frame: {frame_idx} / {total_frames}"
        (tw, th), _ = cv2.getTextSize(frame_text, FONT, FONT_SCALE, FONT_THICKNESS)
        frame_x     = img_w - tw - PADDING
        frame_y     = th + PADDING
        draw_text_with_shadow(display, frame_text, (frame_x, frame_y))

        # ── Time display — below frame counter ───────────────────────────────
        def frames_to_mmssms(frame_num, fps_val):
            total_ms  = int((frame_num / fps_val) * 1000)
            mins      = total_ms // 60000
            secs      = (total_ms % 60000) // 1000
            ms        = total_ms % 1000
            return f"{mins:02d}:{secs:02d}.{ms:03d}"

        cur_time_str   = frames_to_mmssms(frame_idx, fps)
        total_time_str = frames_to_mmssms(total_frames, fps)
        time_text      = f"{cur_time_str} / {total_time_str}"
        (ttw, tth), _  = cv2.getTextSize(time_text, FONT, FONT_SCALE, FONT_THICKNESS)
        time_x         = img_w - ttw - PADDING
        time_y         = frame_y + tth + PADDING
        draw_text_with_shadow(display, time_text, (time_x, time_y))

        # ── Cursor coordinates — below time display ───────────────────────────
        if cursor_in_bounds:
            coord_text  = f"X: {mouse_x}  Y: {mouse_y}"
            (cw, ch), _ = cv2.getTextSize(coord_text, FONT, FONT_SCALE, FONT_THICKNESS)
            coord_x     = img_w - cw - PADDING
            coord_y     = time_y + ch + PADDING
            draw_text_with_shadow(display, coord_text, (coord_x, coord_y))

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1 if paused else delay) & 0xFF
        if key in (ord('q'), 27):       # Q or ESC → quit
            break
        elif key == ord(' '):           # SPACE → toggle pause
            paused = not paused
            print("Paused." if paused else "Resumed.")
        elif key == 81 or key == 2:     # Left arrow → step back one frame
            paused    = True
            target    = max(0, frame_idx - 2)  # -2 because read() already advanced by 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if ret:
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        elif key == 83 or key == 3:     # Right arrow → step forward one frame
            paused    = True
            ret, frame = cap.read()
            if ret:
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                # End of video — stay on last frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    video_path = r"/home/mohammed/Desktop/Mohammed/UPM/Term 8/AI 492 - Capstone/Haram collected vids/Original/Vid_2.mp4"
    csv_dir = r"/home/mohammed/Desktop/Mohammed/UPM/Term 8/AI 492 - Capstone/Haram collected vids/YOLO_Coordinates/v8"

    play_video(video_path, csv_dir)