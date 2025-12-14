#本代码主要由gpt生成
"""
magic_pipeline_optimized.py

目标：将摄像头采集、MediaPipe 手势检测（耗时）与渲染解耦，避免等待/阻塞，
实现更低延迟、更流畅的实时效果（即便 CPU/GPU 占用并不高时也不卡顿）。

使用说明：
- 修改 MAGIC_VIDEO_PATH 为你的本地魔法素材路径。
- 调整 DETECT_WIDTH / DETECT_INTERVAL /PRELOAD_LIMIT 等参数以平衡延迟与精度。
- 如果你的机器较弱，可适当关闭后处理或减小检测频率。
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import time
from multiprocessing import Process, Queue, set_start_method
import multiprocessing
import threading
from collections import deque

# =====================================================
# ============== 配置区（请根据机器情况调整） ===========
# =====================================================
MAGIC_VIDEO_PATH = r"D:\pycharm  study\yolo\第三期，魔法阵分享【特效素材分享】.mp4"
CAMERA_ID = 0

# 视频时间轴（秒 -> 帧）
FPS = 30
GEN_END_SEC = 2.2
LOOP_END_SEC = 5.2
GEN_END = int(GEN_END_SEC * FPS)
LOOP_END = int(LOOP_END_SEC * FPS)

# 绿幕 HSV（如不合适请调整）
HSV_LOWER = np.array([35, 43, 46], dtype=np.uint8)
HSV_UPPER = np.array([77, 255, 255], dtype=np.uint8)

# 预加载：只加载到 LOOP_END + margin，避免预加载整个长视频
PRELOAD_MAGIC = True
PRELOAD_LIMIT = LOOP_END + 60

# 手势检测参数（性能关键）
DETECT_WIDTH = 320            # 送到检测进程的小帧宽度（越小越快）
DETECT_INTERVAL = 2           # 主循环多少帧送一次检测 (1 = 每帧)
MODEL_COMPLEXITY = 0          # MediaPipe Hands 模型复杂度（0/1/2，0 最快）

# smoothing / voting for stable detection (short window)
SMOOTH_WINDOW = 5

# 是否启用电影级后处理（会消耗 CPU）
ENABLE_CINEMATIC_POST = True

# 控制渲染开销: 在 STATE_LOOP 才使用部分 heavy effect
USE_HEAVY_EFFECTS_ONLY_IN_LOOP = True

# 显示窗口名字
WINDOW_NAME = "Magic Circle AR - Pipeline Optimized"

# =====================================================
# ============== 辅助类/函数 ===========================
# =====================================================
class VideoState:
    STATE_IDLE = 0
    STATE_GEN  = 1
    STATE_LOOP = 2
    STATE_END  = 3

    def __init__(self, path, preload=True, preload_limit=PRELOAD_LIMIT):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open magic video: {path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.preload = preload
        self.frames = []
        self.idx = 0
        self.state = self.STATE_IDLE
        self.preload_limit = min(preload_limit, self.frame_count) if self.frame_count > 0 else preload_limit
        if self.preload:
            self._preload_limited()

    def _preload_limited(self):
        self.frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        c = 0
        while True:
            ret, f = self.cap.read()
            if not ret:
                break
            self.frames.append(f.copy())
            c += 1
            if c >= self.preload_limit:
                break
        if len(self.frames) == 0:
            self.preload = False

    def reset(self):
        if self.preload:
            self.idx = 0
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def set_state(self, s):
        if s == self.state:
            return
        self.state = s
        if s == self.STATE_IDLE:
            self.reset()
        elif s == self.STATE_GEN:
            if self.preload:
                self.idx = 0
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif s == self.STATE_LOOP:
            if self.preload:
                self.idx = min(GEN_END, len(self.frames)-1)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, GEN_END)
        elif s == self.STATE_END:
            if self.preload:
                self.idx = min(LOOP_END, len(self.frames)-1)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, LOOP_END)

    def read_frame(self):
        if self.preload:
            if self.idx >= len(self.frames):
                return None
            f = self.frames[self.idx]
            # advance according to state
            if self.state == self.STATE_GEN:
                if self.idx < GEN_END and self.idx < len(self.frames)-1:
                    self.idx += 1
                else:
                    self.idx = min(GEN_END, len(self.frames)-1)
            elif self.state == self.STATE_LOOP:
                self.idx += 1
                if self.idx >= LOOP_END:
                    self.idx = GEN_END
            elif self.state == self.STATE_END:
                self.idx += 1
                if self.idx >= len(self.frames):
                    self.set_state(self.STATE_IDLE)
                    return None
            else:
                return None
            return f.copy()
        else:
            ret, f = self.cap.read()
            if not ret:
                return None
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if self.state == self.STATE_GEN:
                if cur >= GEN_END:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, GEN_END)
            elif self.state == self.STATE_LOOP:
                if cur >= LOOP_END:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, GEN_END)
            elif self.state == self.STATE_END:
                if cur >= self.frame_count - 1:
                    self.set_state(self.STATE_IDLE)
                    return None
            return f

# chroma key mask (float alpha 0..1)
def chroma_key_alpha(magic_bgr):
    hsv = cv2.cvtColor(magic_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)  # green->255
    mask = cv2.bitwise_not(mask)  # foreground->255
    mask = cv2.GaussianBlur(mask, (11,11), 0)
    return (mask.astype(np.float32) / 255.0)

def additive_overlay(bg, fg, alpha, center):
    """
    在 bg 上以 additive 方式叠加 fg（fg 和 alpha 已裁剪至合适大小）
    center: (cx,cy) 背景像素中心坐标
    """
    h_bg, w_bg = bg.shape[:2]
    h_fg, w_fg = fg.shape[:2]
    cx, cy = center
    x1 = int(cx - w_fg//2); y1 = int(cy - h_fg//2); x2 = x1 + w_fg; y2 = y1 + h_fg
    sx = max(0, -x1); sy = max(0, -y1)
    ex = w_fg - max(0, x2 - w_bg); ey = h_fg - max(0, y2 - h_bg)
    if sx >= ex or sy >= ey:
        return bg
    fg_crop = fg[sy:ey, sx:ex].astype(np.float32) / 255.0
    alpha_crop = alpha[sy:ey, sx:ex][:, :, np.newaxis]  # (h,w,1)
    roi_x1 = max(0, x1); roi_y1 = max(0, y1)
    roi_x2 = roi_x1 + (ex - sx); roi_y2 = roi_y1 + (ey - sy)
    roi = bg[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32) / 255.0
    blended = np.clip(roi + fg_crop * alpha_crop, 0.0, 1.0)
    bg[roi_y1:roi_y2, roi_x1:roi_x2] = (blended * 255.0).astype(np.uint8)
    return bg

# =====================================================
# ============== 手势检测进程 =========================
# =====================================================
def detector_process(in_q: Queue, out_q: Queue, detect_width=DETECT_WIDTH, model_complexity=MODEL_COMPLEXITY):
    """
    运行在单独进程中。
    in_q: 接收 (frame_id:int, small_bgr ndarray) 或 ('STOP',)
    out_q: 发送 (frame_id:int, gesture_state:int, palm_norm_x:float, palm_norm_y:float)
    gesture_state: VideoState.STATE_*
    """
    mp_hands = mp.solutions.hands
    hands_local = mp_hands.Hands(static_image_mode=False,
                                 max_num_hands=1,
                                 model_complexity=model_complexity,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
    try:
        while True:
            item = in_q.get()
            if item is None:
                break
            if item == 'STOP':
                break
            frame_id, small_bgr = item
            # convert BGR->RGB for mediapipe
            small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
            res = hands_local.process(small_rgb)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                # compute normalized palm center using landmark 9 (middle_mcp)
                palm_x = lm.landmark[9].x  # normalized in small frame
                palm_y = lm.landmark[9].y
                # robust open-palm detection (scale-normalized)
                state = gesture_from_landmarks(lm)
                out_q.put((frame_id, state, float(palm_x), float(palm_y)))
            else:
                out_q.put((frame_id, VideoState.STATE_IDLE, 0.5, 0.5))
    except KeyboardInterrupt:
        pass
    finally:
        hands_local.close()

# gesture heuristic (same logic as earlier robust)
def gesture_from_landmarks(landmarks):
    try:
        lm = landmarks.landmark
    except Exception:
        return VideoState.STATE_IDLE
    def nd(a,b):
        return math.hypot(a.x - b.x, a.y - b.y)
    try:
        hand_scale = nd(lm[0], lm[9])
        if hand_scale < 1e-5:
            hand_scale = nd(lm[5], lm[17])
            if hand_scale < 1e-5:
                hand_scale = 0.1
    except Exception:
        hand_scale = 0.1
    px = (lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 4.0
    py = (lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 4.0
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    extended = []
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        try:
            tip = lm[tip_idx]; pip = lm[pip_idx]
            condA = (tip.y < pip.y - 0.015)  # small margin
            dist_tip_palm = math.hypot(tip.x - px, tip.y - py)
            condB = (dist_tip_palm > 0.95 * hand_scale)
            extended.append(int(condA or condB))
        except Exception:
            extended.append(0)
    ext_count = sum(extended)
    try:
        thumb_ext = abs(lm[4].x - lm[3].x) > 0.03
    except Exception:
        thumb_ext = False
    if ext_count >= 4 and thumb_ext:
        return VideoState.STATE_END
    if ext_count == 0:
        return VideoState.STATE_LOOP
    if extended[0] == 1 and all(e == 0 for e in extended[1:]):
        return VideoState.STATE_GEN
    return VideoState.STATE_IDLE

# =====================================================
# ============== 主线程：采集 + 渲染 ===================
# =====================================================
def main():
    # Windows multiprocess start method safe
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise IOError("Cannot open camera")

    ret, first = cap.read()
    if not ret:
        raise IOError("Cannot read from camera")
    first = cv2.flip(first, 1)
    H, W = first.shape[:2]

    # Preload magic frames (limited)
    video_state = VideoState(MAGIC_VIDEO_PATH, preload=PRELOAD_MAGIC, preload_limit=PRELOAD_LIMIT)
    print(f"Preloaded magic frames: {len(video_state.frames) if video_state.preload else 0}, total frames in file: {video_state.frame_count}")

    # Queues for detector
    in_q = Queue(maxsize=3)
    out_q = Queue(maxsize=3)
    detector = Process(target=detector_process, args=(in_q, out_q, DETECT_WIDTH, MODEL_COMPLEXITY))
    detector.daemon = True
    detector.start()

    # Shared variables for detection result
    latest_detection = {'frame_id': -1, 'state': VideoState.STATE_IDLE, 'px': 0.5, 'py': 0.5}
    detection_history = deque(maxlen=SMOOTH_WINDOW)

    # Camera capture thread (pushes latest frame into simple holder)
    frame_holder = {'frame': first, 'id': 0}
    stop_flag = {'stop': False}

    def capture_thread():
        fid = 0
        while not stop_flag['stop']:
            ret, f = cap.read()
            if not ret:
                continue
            f = cv2.flip(f, 1)
            frame_holder['frame'] = f
            frame_holder['id'] = fid
            fid += 1
            time.sleep(0.001)  # yield a bit

    t = threading.Thread(target=capture_thread, daemon=True)
    t.start()

    # Main rendering loop
    frame_counter = 0
    last_sent_detect_id = -1
    accum = None  # for light motion blend in loop
    prev_state = VideoState.STATE_IDLE

    print("Press ESC to quit. Running...")

    try:
        while True:
            # get latest frame (non-blocking)
            curr_frame = frame_holder['frame']
            curr_id = frame_holder['id']

            # prepare small frame for detection and send periodically
            if (frame_counter % DETECT_INTERVAL == 0) and (not in_q.full()):
                scale = DETECT_WIDTH / float(W)
                small_h = max(8, int(H * scale))
                small = cv2.resize(curr_frame, (DETECT_WIDTH, small_h), interpolation=cv2.INTER_LINEAR)
                try:
                    in_q.put_nowait((curr_id, small))
                    last_sent_detect_id = curr_id
                except:
                    pass

            # collect any pending detection outputs (drain queue)
            while not out_q.empty():
                try:
                    fid, st, px, py = out_q.get_nowait()
                    # use most recent result (frame id closest to current)
                    latest_detection.update({'frame_id': fid, 'state': st, 'px': px, 'py': py})
                except:
                    break

            # smoothing via short history
            detection_history.append(latest_detection['state'])
            # majority vote
            counts = {}
            for s in detection_history:
                counts[s] = counts.get(s, 0) + 1
            smoothed_state = max(counts.items(), key=lambda x: x[1])[0]

            # compute palm pixel coords in current frame from normalized
            px = int(latest_detection['px'] * W)
            py = int(latest_detection['py'] * H)
            hand_center = (px, py)

            # set video state
            video_state.set_state(smoothed_state)

            # if state changed, reset accum to prevent trailing residuals
            if smoothed_state != prev_state:
                accum = None

            # get magic frame if available and overlay
            out = curr_frame.copy()
            magic_frame = video_state.read_frame()
            if magic_frame is not None:
                # resize magic to scale factor (small) to reduce overlay cost
                scale_factor = 0.7
                mh = max(8, int(magic_frame.shape[0] * scale_factor))
                mw = max(8, int(magic_frame.shape[1] * scale_factor))
                try:
                    mf_rs = cv2.resize(magic_frame, (mw, mh), interpolation=cv2.INTER_AREA)
                    alpha = chroma_key_alpha(mf_rs)
                    out = additive_overlay(out, mf_rs, alpha, hand_center)
                except Exception as e:
                    # overlay must not crash pipeline
                    print("Overlay exception:", e)

            # light temporal blend only in loop to add subtle motion, but keep small weight
            if (smoothed_state == VideoState.STATE_LOOP) and ENABLE_CINEMATIC_POST:
                if accum is None:
                    accum = out.astype(np.float32) / 255.0
                else:
                    w = 0.6
                    accum = accum * w + (out.astype(np.float32) / 255.0) * (1.0 - w)
                out_disp = np.clip(accum * 255.0, 0, 255).astype(np.uint8)
            else:
                out_disp = out

            # optional cinematic post (cheap)
            if ENABLE_CINEMATIC_POST:
                if not USE_HEAVY_EFFECTS_ONLY_IN_LOOP or smoothed_state == VideoState.STATE_LOOP:
                    # small vignette + tiny film grain + color grade cheap
                    out_disp = color_grade_simple(out_disp)
                    out_disp = small_vignette(out_disp, strength=0.35)
                    out_disp = tiny_film_grain(out_disp, intensity=0.02)

            cv2.imshow(WINDOW_NAME, out_disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            prev_state = smoothed_state
            frame_counter += 1
    finally:
        # cleanup
        stop_flag['stop'] = True
        cap.release()
        # notify detector to stop
        try:
            in_q.put_nowait('STOP')
        except:
            pass
        detector.join(timeout=1.0)
        cv2.destroyAllWindows()

# cheap postprocess helpers (very lightweight)
def color_grade_simple(frame):
    f = frame.astype(np.float32) / 255.0
    b, g, r = cv2.split(f)
    r = np.clip(r * 0.98 + 0.01 * (1 - r), 0.0, 1.0)
    b = np.clip(b * 1.02, 0.0, 1.0)
    out = cv2.merge([b,g,r])
    return (np.clip(out,0,1)*255).astype(np.uint8)

def small_vignette(frame, strength=0.35):
    h, w = frame.shape[:2]
    X = np.linspace(-1,1,w); Y = np.linspace(-1,1,h)
    xv, yv = np.meshgrid(X,Y)
    d = np.sqrt(xv**2 + yv**2)
    m = 1.0 - strength * (d**2)
    m = np.clip(m, 0.0, 1.0)[:,:,np.newaxis]
    out = frame.astype(np.float32)/255.0 * m
    return (np.clip(out,0,1)*255).astype(np.uint8)

def tiny_film_grain(frame, intensity=0.02):
    if intensity <= 0:
        return frame
    h,w = frame.shape[:2]
    noise = np.random.randn(h,w).astype(np.float32)
    # normalize
    mn, mx = noise.min(), noise.max()
    if mx - mn < 1e-6:
        norm = np.zeros_like(noise)
    else:
        norm = (noise - mn) / (mx - mn)
    noise_u8 = (norm * 255).astype(np.uint8)
    noise_u8 = cv2.GaussianBlur(noise_u8, (5,5), 0)
    noise_f = (noise_u8.astype(np.float32)/255.0 - 0.5) * intensity
    noise_f = noise_f[:,:,np.newaxis]
    f = frame.astype(np.float32)/255.0
    f = np.clip(f + noise_f, 0.0, 1.0)
    return (f*255.0).astype(np.uint8)

# =====================================================
# ============== 程序入口 ==============================
# =====================================================
if __name__ == "__main__":
    main()
