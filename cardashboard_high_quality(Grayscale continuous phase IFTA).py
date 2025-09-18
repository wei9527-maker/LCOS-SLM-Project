import os, time, threading
from collections import deque
import numpy as np
import cv2
import cupy as cp
import tkinter as tk
from Car_Speedometer_Project_Version import CarSimGaugeOnly
import matplotlib.pyplot as plt
import mss


try:
    import ctypes
    ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)  
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

GAUGE_MONITOR_INDEX = 1
SLM_MONITOR_INDEX   = 0


CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(CUDA_BIN)
else:
    os.environ["PATH"] = CUDA_BIN + ";" + os.environ.get("PATH", "")

W, H = 1920, 1080
MARGIN_RATIO = 0.01
ZOOM = 1.0
SLM_MIRROR = "h"

lam = 532e-6
PS  = 12.17 / W
z1  = 1700
k   = 2*np.pi/lam
level = 256
a = 0.0/(2*PS)
MAX_ITER = 10
RMS_STOP = 0.6
FPS = 10
TARGET_HZ = 10   

STRETCH_X = 1.70  
STRETCH_Y = 1.00   

try:
    from screeninfo import get_monitors
    _MONITORS = get_monitors()
except Exception as e:
    print("⚠️ 無法載入 screeninfo，將以單螢幕模式移動視窗。", e)
    _MONITORS = None

def get_monitor_rect(index: int):
    if _MONITORS and 0 <= index < len(_MONITORS):
        m = _MONITORS[index]
        return (m.x, m.y, m.width, m.height)
    tmp = tk.Tk(); tmp.withdraw()
    w = tmp.winfo_screenwidth(); h = tmp.winfo_screenheight()
    tmp.destroy()
    return (0, 0, w, h)

def grab_monitor_to_canvas_mss_ctx(ctx, mss_index: int, W=1920, H=1080, margin_ratio=0.0, zoom=1.0):
    mons = ctx.monitors
    m = mons[mss_index] if 1 <= mss_index < len(mons) else mons[1]
    shot  = ctx.grab(m) 
    frame = np.frombuffer(shot.rgb, dtype=np.uint8).reshape(shot.height, shot.width, 3)[:, :, ::-1]  # to BGR

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    Hc, Wc = H, W
    canvas = np.zeros((Hc, Wc), np.uint8)

    h, w = bw.shape[:2]
    mW = int(Wc*(1-2*margin_ratio))
    mH = int(Hc*(1-2*margin_ratio))

    s = min(mW/w, mH/h) * float(zoom)
    new_w = max(1, int(round(w*s)))
    new_h = max(1, int(round(h*s)))
    resized = cv2.resize(bw, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)

    ox = (Wc - mW)//2
    oy = (Hc - mH)//2
    if new_w > mW:
        x0 = (new_w - mW)//2
        resized = resized[:, x0:x0+mW]; new_w = mW
    if new_h > mH:
        y0 = (new_h - mH)//2
        resized = resized[y0:y0+mH, :]; new_h = mH

    xoff = ox + (mW - new_w)//2
    yoff = oy + (mH - new_h)//2
    canvas[yoff:yoff+new_h, xoff:xoff+new_w] = resized
    return canvas

def aniso_scale_center(img, sx: float, sy: float):
    """以影像中心為原點做 X/Y 縮放，超出裁掉、缺口補 0。"""
    h, w = img.shape[:2]
    M = np.float32([
        [sx, 0, (1.0 - sx) * w * 0.5],
        [0, sy, (1.0 - sy) * h * 0.5]
    ])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=0)


def precompute_fresnel(W, H, PS, z1, lam, k, a):
    y_pix, x_pix = cp.ogrid[:H, :W]
    cx, cy = (W-1)/2, (H-1)/2
    X = (x_pix - cx) * PS
    Y = (y_pix - cy) * PS
    R = cp.sqrt(X**2 + Y**2)
    A1  = cp.exp(1j*k*R**2/(2*z1)) * cp.exp(1j*k*z1) / (1j*lam*z1)
    h1  = cp.exp(1j*k*R**2/(2*z1))
    B1  = cp.exp(-1j*k*R**2/(2*z1)) * cp.exp(-1j*k*z1) / (1j*lam*z1)
    h11 = cp.exp(-1j*k*R**2/(2*z1))
    carrier = cp.exp(1j * 2 * cp.pi * a * Y)
    return A1, h1, B1, h11, carrier

A1, h1, B1, h11, carrier = precompute_fresnel(W, H, PS, z1, lam, k, a)

def run_ifta(Upic, O_prev, max_iter=MAX_ITER, rms_stop=RMS_STOP, level=level):
    O = O_prev
    last_phase = None
    inner_rms_val = np.nan
    iters = 0
    for i in range(1, max_iter+1):
        O12 = A1 * cp.fft.fft2(O * h1, norm="ortho")
        O13 = Upic * cp.exp(1j * cp.angle(O12))
        O14 = B1 * cp.fft.ifft2(O13 * h11, norm="ortho")
        q = cp.round(((cp.angle(O14) + cp.pi) / (2*cp.pi) * level), 0)
        O = cp.exp(1j * q * (2*cp.pi) / level)

        cur_phase = cp.angle(O)
        if last_phase is not None:
            inner_rms = cp.sqrt(cp.mean((cur_phase - last_phase)**2))
            inner_rms_val = float(inner_rms.get())
            if inner_rms_val < rms_stop:
                iters = i
                break
        last_phase = cur_phase
        iters = i

    phase_u8 = (cp.angle(O*carrier) % (2*cp.pi)) / (2*cp.pi) * 255.0
    return O, phase_u8.astype(cp.uint8), iters, inner_rms_val

def create_positioned_window(win_name: str, mon_rect, fullscreen=True, size=None):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    x, y, w, h = mon_rect
    if size is None:
        cv2.resizeWindow(win_name, w, h)
    else:
        cv2.resizeWindow(win_name, size[0], size[1])
    cv2.moveWindow(win_name, x, y)
    if fullscreen:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def apply_mirror(img_np, mode):
    if mode is None: return img_np
    if mode == "h":  return cv2.flip(img_np, 1)
    if mode == "v":  return cv2.flip(img_np, 0)
    if mode in ("hv","vh"): return cv2.flip(img_np, -1)
    return img_np


def capture_worker(stop_event, mss_index, W, H, margin_ratio, zoom, out_deque):
    ctx = mss.mss()  
    while not stop_event.is_set():
        frame = grab_monitor_to_canvas_mss_ctx(ctx, mss_index, W, H, margin_ratio, zoom)
        if len(out_deque) == out_deque.maxlen:
            out_deque.popleft()
        out_deque.append(frame)

def hologram_worker(stop_event, capture_mss_index, slm_monitor_rect, metrics, frame_q):
    """
    metrics: dict
      - frame_times, gpu_times, iters, inner_rms, inter_rms
    """
    win_name = "SLM"
    create_positioned_window(win_name, slm_monitor_rect, fullscreen=True)


    O_prev = cp.exp(1j * 2 * cp.pi * cp.random.rand(H, W).astype(cp.float32)).astype(cp.complex64)
    last_final_phase = None
    t_next = time.time()
    frame_idx = 1


    d_target = cp.empty((H, W), dtype=cp.float32)   
    shift_hw = (H // 2, W // 2)                     

    while not stop_event.is_set():
        t0 = time.perf_counter()

        if not frame_q:
            time.sleep(0.001)
            continue


        canvas = frame_q[-1] 
        if STRETCH_X != 1.0 or STRETCH_Y != 1.0:
            canvas = aniso_scale_center(canvas, STRETCH_X, STRETCH_Y)
        d_target.set(canvas.astype(np.float32) / 255.0)  

        targ_shift = cp.roll(d_target, shift_hw, axis=(0, 1))
        Upic = cp.sqrt(targ_shift + cp.float32(1e-8))

        gpu_t0 = time.perf_counter()
        O_prev, phase_u8, iters, inner_rms_val = run_ifta(Upic, O_prev)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.perf_counter() - gpu_t0


        cur_final_phase = cp.angle(O_prev)
        if last_final_phase is not None:
            inter_rms = cp.sqrt(cp.mean((cur_final_phase - last_final_phase)**2))
            inter_rms_val = float(inter_rms.get())
        else:
            inter_rms_val = float('nan')
        last_final_phase = cur_final_phase


        frame_np = apply_mirror(cp.asnumpy(phase_u8), SLM_MIRROR)
        cv2.imshow(win_name, frame_np)

        frame_time = time.perf_counter() - t0
        hz = (1.0 / frame_time) if frame_time > 0 else float('inf')


        print(f"Frame {frame_idx:04d} | 收斂於 {iters} 次迭代 (RMS: {inner_rms_val:.4f})")
        print(f"Frame {frame_idx:04d} | RMS Phase Diff: {inter_rms_val:.6f}")
        print(f"Frame {frame_idx:04d} | GPU time: {gpu_time:.3f}s | {hz:.1f} Hz")

        metrics["frame_times"].append(frame_time)
        metrics["gpu_times"].append(gpu_time)
        metrics["iters"].append(iters)
        metrics["inner_rms"].append(inner_rms_val)
        metrics["inter_rms"].append(inter_rms_val)

        frame_idx += 1

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            stop_event.set(); break

        t_next += 1.0 / FPS
        while time.time() < t_next and not stop_event.is_set():
            time.sleep(0.001)

    cv2.destroyAllWindows()



def main():
    gauge_rect = get_monitor_rect(GAUGE_MONITOR_INDEX)
    slm_rect   = get_monitor_rect(SLM_MONITOR_INDEX)
    print(f"🖥️  擷取螢幕(index={GAUGE_MONITOR_INDEX}): {gauge_rect}")
    print(f"🖥️  SLM 螢幕(index={SLM_MONITOR_INDEX}): {slm_rect}")

    root = tk.Tk()
    app  = CarSimGaugeOnly(root)
    gx, gy, gw, gh = gauge_rect
    root.overrideredirect(True)
    root.geometry(f"{gw}x{gh}+{gx}+{gy}")
    root.update_idletasks()

    capture_mss_index = GAUGE_MONITOR_INDEX + 1

    metrics = {k: [] for k in ["frame_times","gpu_times","iters","inner_rms","inter_rms"]}
    frame_q = deque(maxlen=1)
    stop_event = threading.Event()

    cap_th = threading.Thread(target=capture_worker,
                              args=(stop_event, capture_mss_index, W, H, MARGIN_RATIO, ZOOM, frame_q),
                              daemon=True)
    cap_th.start()

    holo_th = threading.Thread(target=hologram_worker,
                               args=(stop_event, capture_mss_index, slm_rect, metrics, frame_q),
                               daemon=True)
    holo_th.start()

    tStart = time.time()
    try:
        root.mainloop()
    finally:
        stop_event.set()
        holo_th.join(timeout=2.0)
        cap_th.join(timeout=2.0)

        tEnd = time.time()
        n = len(metrics["frame_times"])
        if n > 0:
            total_time = tEnd - tStart
            avg_fps = n / total_time if total_time > 0 else float('inf')
            print(f"\n✅ 結束：共 {n} 幀，總耗時 {total_time:.2f}s，平均 {avg_fps:.2f} Hz")

            x = np.arange(1, n+1)
            plt.figure(figsize=(10, 5))
            plt.plot(x, metrics["frame_times"], label="Frame Time (s)", linewidth=1)
            plt.plot(x, metrics["gpu_times"], label="GPU Time (s)", linewidth=1)
            plt.hlines(1.0/FPS, 1, n, linestyles="dashed", label=f"Target {FPS} FPS ({1.0/FPS:.3f}s)")
            plt.xlabel("Frame"); plt.ylabel("Time (s)")
            plt.title("Per-Frame Computation Time")
            plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
            plt.savefig("frame_times.png", dpi=150); plt.show()
        else:
            print("（沒有收集到幀資料）")

if __name__ == "__main__":
    main()

















