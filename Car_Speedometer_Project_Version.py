import tkinter as tk
import math, time


MASS_KG = 1450.0
WHEEL_RADIUS_M = 0.30
FRONTAL_AREA_M2 = 2.2
CD = 0.29
C_RR = 0.012
AIR_RHO = 1.225
DRIVETRAIN_EFF = 0.90
MU_TIRE = 1.0
MAX_BRAKE_DECEL = 7.0

IDLE_RPM = 750.0
FINAL_DRIVE = 3.42
GEAR_RATIOS = [3.60, 2.19, 1.41, 1.12, 0.86, 0.67]
SHIFT_UP_RPM = 6200.0
SHIFT_DOWN_RPM = 1500.0
SHIFT_COOLDOWN = 0.35

TORQUE_CURVE = [
    (700,  90),(1200,120),(2000,160),(2800,190),
    (3500,220),(4500,250),(5200,245),(5800,230),(6500,200),
]

THROTTLE_RISE_PER_S = 2.5
THROTTLE_FALL_PER_S = 1.8
IDLE_THROTTLE = 0.08
CREEP_TORQUE = 40.0
ENGINE_BRAKE_K = 45.0

WIN_W, WIN_H = 1920, 1080     
SPEED_MAX_KMH = 240.0
UPDATE_HZ = 60

def clamp(x, lo, hi): return lo if x < lo else (hi if x > hi else x)
def lerp(a, b, t): return a + (b - a) * t

def interp_curve(x, pts):
    if x <= pts[0][0]: return pts[0][1]
    if x >= pts[-1][0]: return pts[-1][1]
    for i in range(len(pts)-1):
        x0,y0 = pts[i]; x1,y1 = pts[i+1]
        if x0 <= x <= x1:
            t=(x-x0)/(x1-x0); return lerp(y0,y1,t)
    return pts[-1][1]

def speed_to_rpm(v_ms, gear_ratio):
    if gear_ratio <= 0: return IDLE_RPM
    wheel_rps  = v_ms / (2*math.pi*WHEEL_RADIUS_M)
    engine_rps = wheel_rps * gear_ratio * FINAL_DRIVE
    return max(engine_rps * 60.0, IDLE_RPM)

def drag_force(v_ms): return 0.5 * AIR_RHO * CD * FRONTAL_AREA_M2 * v_ms*v_ms
def rolling_resistance(): return C_RR * MASS_KG * 9.81

class CarSimGaugeOnly:
    def __init__(self, root):
        self.root = root
        root.title("Speedometer  ↑油門 / Space煞車 / R重置 / Esc離開")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        x = (sw - WIN_W) // 2
        y = (sh - WIN_H) // 2
        root.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")
        root.configure(bg="#0e1116")

        try:
            root.state('zoomed')                    
        except tk.TclError:
            pass        


        self.canvas = tk.Canvas(root, bg="#0e1116", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)


        self.v_ms = 0.0
        self.gear = 1
        self.throttle = 0.0
        self.braking = 0.0
        self.shift_timer = 0.0
        self.last_time = time.time()
        self.want_throttle = IDLE_THROTTLE
        self.key_throttle = False
        self.key_brake = False


        self.start_deg = 225.0
        self.end_deg   = -45.0


        self._build_ui()


        root.bind("<Configure>", self._on_resize)
        root.bind("<KeyPress>", self._on_key_down)
        root.bind("<KeyRelease>", self._on_key_up)


        self.update_loop()

    def _metrics(self):
        w = self.canvas.winfo_width()  or WIN_W
        h = self.canvas.winfo_height() or WIN_H

        r  = 0.47 * min(w, h)
        cx = w * 0.5
        cy = h * 0.52
        fs_speed = max(16, int(min(w,h) * 0.10))
        lw_major = max(2, int(min(w,h) * 0.010))
        lw_minor = max(1, int(min(w,h) * 0.006))
        lw_needle= max(4, int(min(w,h) * 0.012))
        return dict(w=w,h=h,cx=cx,cy=cy,r=r,
                    fs_speed=fs_speed,lw_major=lw_major,lw_minor=lw_minor,lw_needle=lw_needle)

    def _build_ui(self):
        self.canvas.delete("all")
        M = self._metrics()
        cx, cy, r = M["cx"], M["cy"], M["r"]


        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline= "#d2d8e6", width=6, fill="#20242c")

        for s in range(0, int(SPEED_MAX_KMH)+1, 20):
            t = s / SPEED_MAX_KMH
            ang = math.radians(lerp(self.start_deg, self.end_deg, t))
            inner = r - (r*0.11 if s % 40 == 0 else r*0.07)
            x1 = cx + math.cos(ang) * inner
            y1 = cy - math.sin(ang) * inner
            x2 = cx + math.cos(ang) * (r - r*0.02)
            y2 = cy - math.sin(ang) * (r - r*0.02)
            self.canvas.create_line(x1, y1, x2, y2,
                                    fill="#b4becc",
                                    width=(M["lw_major"] if s % 40 == 0 else M["lw_minor"]))
            if s % 40 == 0:
                tx = cx + math.cos(ang) * (inner - r*0.12)
                ty = cy - math.sin(ang) * (inner - r*0.12)
                self.canvas.create_text(tx, ty, text=str(s), fill="#d2d8e6",
                                        font=("Consolas", max(60, int(r*0.10*0.25))))


        self.needle = self.canvas.create_line(cx, cy, cx, cy-(r - r*0.16),
                                              width=M["lw_needle"], fill="#ff5050", capstyle="round")

        hub_r = r*0.035
        self.hub = self.canvas.create_oval(cx-hub_r, cy-hub_r, cx+hub_r, cy+hub_r,
                                           fill="#dc5050", outline="")\

        self.speed_text = self.canvas.create_text(cx, cy + r*0.30, text="0km/h",
                                                  fill="#ebeef5",
                                                  font=("Consolas", M["fs_speed"], "bold"))

        self._geom = dict(cx=cx, cy=cy, r=r)

    def _on_resize(self, _e): self._build_ui()

    def _on_key_down(self, ev):
        k = ev.keysym
        if k == "Up": self.key_throttle = True
        elif k == "space": self.key_brake = True
        elif k in ("r","R"): self._reset()
        elif k == "Escape": self.root.destroy()

    def _on_key_up(self, ev):
        k = ev.keysym
        if k == "Up": self.key_throttle = False
        elif k == "space": self.key_brake = False

    def _reset(self):
        self.v_ms = 0.0; self.gear = 1; self.throttle = 0.0
        self.braking = 0.0; self.shift_timer = 0.0
        self.want_throttle = IDLE_THROTTLE

    def update_loop(self):
        now = time.time()
        dt = clamp(now - self.last_time, 0.0001, 0.05)
        self.last_time = now

        self.want_throttle = 1.0 if self.key_throttle else IDLE_THROTTLE
        if self.want_throttle > self.throttle:
            self.throttle = clamp(self.throttle + THROTTLE_RISE_PER_S*dt, 0.0, self.want_throttle)
        else:
            self.throttle = clamp(self.throttle - THROTTLE_FALL_PER_S*dt, IDLE_THROTTLE, 1.0)
        self.braking = 1.0 if self.key_brake else 0.0

        ratio = GEAR_RATIOS[self.gear-1]
        rpm = speed_to_rpm(self.v_ms, ratio)

        if self.shift_timer > 0.0: self.shift_timer -= dt
        if self.shift_timer <= 0.0 and self.gear < len(GEAR_RATIOS):
            if rpm > SHIFT_UP_RPM and self.throttle > 0.25:
                self.gear += 1; self.shift_timer = SHIFT_COOLDOWN; ratio = GEAR_RATIOS[self.gear-1]; rpm = speed_to_rpm(self.v_ms, ratio)
        if self.shift_timer <= 0.0 and self.gear > 1:
            if rpm < SHIFT_DOWN_RPM and (self.throttle > 0.35 or self.v_ms < 6.0):
                self.gear -= 1; self.shift_timer = SHIFT_COOLDOWN; ratio = GEAR_RATIOS[self.gear-1]; rpm = speed_to_rpm(self.v_ms, ratio)

        torque = interp_curve(rpm, TORQUE_CURVE) * self.throttle
        if self.gear == 1 and self.braking == 0.0 and self.throttle <= IDLE_THROTTLE + 0.02:
            torque += CREEP_TORQUE
        if self.throttle <= IDLE_THROTTLE + 0.01:
            torque -= (ENGINE_BRAKE_K * max(0.0, rpm - IDLE_RPM) / 1000.0)

        drive_force = 0.0 if self.shift_timer > 0.0 else clamp(
            torque * ratio * FINAL_DRIVE * DRIVETRAIN_EFF / WHEEL_RADIUS_M,
            -MU_TIRE*MASS_KG*9.81, MU_TIRE*MASS_KG*9.81
        )
        F_net = drive_force - drag_force(self.v_ms) - rolling_resistance() \
                - (MASS_KG*MAX_BRAKE_DECEL*self.braking if self.braking>0 else 0.0)
        a = F_net / MASS_KG

        self.v_ms = max(0.0, self.v_ms + a*dt)
        if self.v_ms < 0.02 and self.throttle <= IDLE_THROTTLE + 0.01 and self.braking == 0.0:
            self.v_ms = 0.0

        G = self._geom; cx, cy, r = G["cx"], G["cy"], G["r"]
        kmh = self.v_ms * 3.6
        t = clamp(kmh / SPEED_MAX_KMH, 0.0, 1.0)
        ang = math.radians(lerp(self.start_deg, self.end_deg, t))
        nx = cx + math.cos(ang) * (r - r*0.16)
        ny = cy - math.sin(ang) * (r - r*0.16)
        self.canvas.coords(self.needle, cx, cy, nx, ny)
        self.canvas.itemconfigure(self.speed_text, text=f"{int(round(kmh))} km/h")

        self.root.after(int(1000/UPDATE_HZ), self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app  = CarSimGaugeOnly(root)
    root.mainloop()













