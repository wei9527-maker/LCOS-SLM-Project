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
REDLINE_RPM = 6500.0
FINAL_DRIVE = 3.42
GEAR_RATIOS = [3.60, 2.19, 1.41, 1.12, 0.86, 0.67]   # 1~6 檔
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
WIN_W, WIN_H = 640, 360
SPEED_MAX_KMH = 240.0
UPDATE_HZ = 60

def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def lerp(a, b, t): return a + (b - a) * t

def interp_curve(x, pts):
    if x <= pts[0][0]: return pts[0][1]
    if x >= pts[-1][0]: return pts[-1][1]
    for i in range(len(pts)-1):
        x0, y0 = pts[i]; x1, y1 = pts[i+1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return lerp(y0, y1, t)
    return pts[-1][1]

def speed_to_rpm(v_ms, gear_ratio):
    if gear_ratio <= 0: return IDLE_RPM
    wheel_rps = v_ms / (2*math.pi*WHEEL_RADIUS_M)
    engine_rps = wheel_rps * gear_ratio * FINAL_DRIVE
    return max(engine_rps * 60.0, IDLE_RPM)

def drag_force(v_ms):
    return 0.5 * AIR_RHO * CD * FRONTAL_AREA_M2 * v_ms*v_ms

def rolling_resistance():
    return C_RR * MASS_KG * 9.81

def place_bottom_right(root, width=None, height=None, margin=20):
        """把視窗放到主螢幕右下角，margin 為邊距像素。"""
        root.update_idletasks()
        if width is None or height is None:
            width  = root.winfo_width()
            height = root.winfo_height()
            if width <= 1 or height <= 1:
                width, height = 640, 360

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = sw - width  - margin
        y = sh - height - margin
        root.geometry(f"{width}x{height}+{x}+{y}")

class CarSimApp:
    def __init__(self, root):
        self.root = root
        root.title("Car Speedometer (Tk)  ↑油門 / Space煞車 / R重置 / Esc離開")
        root.geometry(f"{WIN_W}x{WIN_H}")
        root.configure(bg="#0e1116")
        root.minsize(480, 270) 

        self.canvas = tk.Canvas(root, bg="#0e1116", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)


        self.v_ms = 0.0
        self.gear = 1
        self.throttle = 0.0
        self.braking = 0.0
        self.shift_timer = 0.0
        self.odometer_km = 0.0
        self.want_throttle = IDLE_THROTTLE
        self.key_throttle = False
        self.key_brake = False
        self.last_time = time.time()

        self.gauge_cx_p = 0.30     
        self.gauge_cy_p = 0.50     
        self.gauge_r_p  = 0.42     
        self.start_deg = 225.0
        self.end_deg   = -45.0

        self.panel_x_p = 0.58      
        self.panel_y_p = 0.06
        self.panel_w_p = 0.38
        self.panel_h_p = 0.84
        self.panel_pad_p = 0.02    

        self.items = {}
        self.fonts = {}

        self._rebuild_ui()

        root.bind("<Configure>", self._on_resize)
        root.bind("<KeyPress>", self.on_key_press)
        root.bind("<KeyRelease>", self.on_key_release)

        self.update_loop()

    def _metrics(self):
        w = self.canvas.winfo_width()  or WIN_W
        h = self.canvas.winfo_height() or WIN_H

        cx = self.gauge_cx_p * w
        cy = self.gauge_cy_p * h
        radius = self.gauge_r_p * h

        rx = self.panel_x_p * w
        ry = self.panel_y_p * h
        rw = self.panel_w_p * w
        rh = self.panel_h_p * h
        pad = self.panel_pad_p * rw

        fs_speed = max(14, int(h * 0.085))     
        fs_h1    = max(12, int(h * 0.045))     
        fs_lbl   = max(10, int(h * 0.030))     
        fs_tip   = max(9,  int(h * 0.022))     

        lw_major = max(1, int(h * 0.005))
        lw_minor = max(1, int(h * 0.003))
        lw_needle= max(3, int(h * 0.007))

        bar_h = max(8, int(h * 0.030))

        return {
            "w": w, "h": h,
            "cx": cx, "cy": cy, "radius": radius,
            "rx": rx, "ry": ry, "rw": rw, "rh": rh, "pad": pad,
            "fs_speed": fs_speed, "fs_h1": fs_h1, "fs_lbl": fs_lbl, "fs_tip": fs_tip,
            "lw_major": lw_major, "lw_minor": lw_minor, "lw_needle": lw_needle,
            "bar_h": bar_h,
        }

    def _build_fonts(self, M):
        self.fonts["speed"] = ("Consolas", M["fs_speed"], "bold")
        self.fonts["h1"]    = ("Consolas", M["fs_h1"], "bold")
        self.fonts["lbl"]   = ("Consolas", M["fs_lbl"])
        self.fonts["tip"]   = ("Consolas", M["fs_tip"])


    def _rebuild_ui(self):
        self.canvas.delete("all")
        M = self._metrics()
        self._build_fonts(M)

        cx, cy, r = M["cx"], M["cy"], M["radius"]


        self.items["gauge_bg"] = self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline="#3c424e", width=6, fill="#20242c"
        )

        ticks = list(range(0, int(SPEED_MAX_KMH)+1, 20))
        for s in ticks:
            t = s / SPEED_MAX_KMH
            ang = math.radians(lerp(self.start_deg, self.end_deg, t))
            inner = r - (r*0.11 if s % 40 == 0 else r*0.07)
            x1 = cx + math.cos(ang) * inner
            y1 = cy - math.sin(ang) * inner
            x2 = cx + math.cos(ang) * (r - r*0.02)
            y2 = cy - math.sin(ang) * (r - r*0.02)
            self.canvas.create_line(
                x1, y1, x2, y2,
                fill="#b4becc",
                width=(M["lw_major"] if s % 40 == 0 else M["lw_minor"])
            )
            if s % 40 == 0:
                tx = cx + math.cos(ang) * (inner - r*0.12)
                ty = cy - math.sin(ang) * (inner - r*0.12)
                self.canvas.create_text(tx, ty, text=str(s), fill="#d2d8e6",
                                        font=self.fonts["lbl"])

        self.items["needle"] = self.canvas.create_line(
            cx, cy, cx, cy - (r - r*0.16),
            width=M["lw_needle"], fill="#ff5050", capstyle="round"
        )
        hub_r = 0.02 * M["h"]
        self.items["hub"] = self.canvas.create_oval(
            cx - hub_r, cy - hub_r, cx + hub_r, cy + hub_r,
            fill="#dc5050", outline=""
        )
        self.items["speed_text"] = self.canvas.create_text(
            cx, cy + 0.15*M["h"], text="0 km/h", fill="#ebeef5", font=self.fonts["speed"]
        )

        rx, ry, rw, rh, pad = M["rx"], M["ry"], M["rw"], M["rh"], M["pad"]
        self.items["panel_bg"] = self.canvas.create_rectangle(
            rx, ry, rx+rw, ry+rh, fill="#1c2028", outline="#3a404c", width=3
        )


        x0, y0 = rx + pad, ry + pad
        self.items["txt_gear"] = self.canvas.create_text(
            x0, y0, text="Gear: 1 / 6", fill="#e6e6f0", anchor="nw", font=self.fonts["h1"]
        )
        self.items["txt_rpm"] = self.canvas.create_text(
            x0, y0 + 1.2*M["fs_h1"], text="RPM : 750", fill="#e6e6f0", anchor="nw", font=self.fonts["h1"]
        )


        bar_left  = rx + pad
        bar_right = rx + rw - pad
        th_y = y0 + 1.2*M["fs_h1"] + 2.2*M["fs_h1"]
        br_y = th_y + 2.2*M["bar_h"]
        self.items["th_label"] = self.canvas.create_text(
            bar_left, th_y - 1.3*M["bar_h"],
            text="Throttle", fill="#dcdce6", anchor="nw", font=self.fonts["lbl"])
        self.items["th_bg"] = self.canvas.create_rectangle(
            bar_left, th_y, bar_right, th_y + M["bar_h"], fill="#5a6473", outline="")
        self.items["th_fg"] = self.canvas.create_rectangle(
            bar_left, th_y, bar_left, th_y + M["bar_h"], fill="#5ab85a", outline="")

        self.items["br_label"] = self.canvas.create_text(
            bar_left, br_y - 1.3*M["bar_h"],
            text="Brake", fill="#dcdce6", anchor="nw", font=self.fonts["lbl"])
        self.items["br_bg"] = self.canvas.create_rectangle(
            bar_left, br_y, bar_right, br_y + M["bar_h"], fill="#5a6473", outline="")
        self.items["br_fg"] = self.canvas.create_rectangle(
            bar_left, br_y, bar_left, br_y + M["bar_h"], fill="#e05a5a", outline="")

        info_y0 = br_y + 2.6*M["bar_h"]
        line_h = 1.6*M["fs_lbl"]
        self.items["txt_driveF"] = self.canvas.create_text(bar_left, info_y0 + 0*line_h,
                                                           text="Drive F: 0 N", fill="#dcdce6", anchor="nw", font=self.fonts["lbl"])
        self.items["txt_dragF"]  = self.canvas.create_text(bar_left, info_y0 + 1*line_h,
                                                           text="Drag  F: 0 N", fill="#dcdce6", anchor="nw", font=self.fonts["lbl"])
        self.items["txt_rollF"]  = self.canvas.create_text(bar_left, info_y0 + 2*line_h,
                                                           text="Roll  F: 0 N", fill="#dcdce6", anchor="nw", font=self.fonts["lbl"])
        self.items["txt_brakeF"] = self.canvas.create_text(bar_left, info_y0 + 3*line_h,
                                                           text="Brake F: 0 N", fill="#dcdce6", anchor="nw", font=self.fonts["lbl"])
        self.items["txt_accel"]  = self.canvas.create_text(bar_left, info_y0 + 4*line_h,
                                                           text="Accel  : 0.00 m/s²", fill="#dcdce6", anchor="nw", font=self.fonts["lbl"])
        self.items["txt_odo"]    = self.canvas.create_text(bar_left, info_y0 + 5*line_h,
                                                           text="Odo    : 0.00 km", fill="#dcdce6", anchor="nw", font=self.fonts["lbl"])


        tip_y = ry + rh - 1.8*M["fs_tip"]
        self.items["tip"] = self.canvas.create_text(
            rx + pad, tip_y,
            text="↑ 油門 / Space 煞車 / R 重置 / Esc 離開",
            fill="#c8cdd8", anchor="nw", font=self.fonts["tip"]
        )


        self.geom = {
            "cx": cx, "cy": cy, "r": r,
            "bar_left": bar_left, "bar_right": bar_right,
            "th_y": th_y, "br_y": br_y, "bar_h": M["bar_h"],
        }


    def _on_resize(self, _event):
        self._rebuild_ui()

    def on_key_press(self, ev):
        k = ev.keysym
        if k == "Up":
            self.key_throttle = True
        elif k == "space":
            self.key_brake = True
        elif k in ("r", "R"):
            self.reset()
        elif k == "Escape":
            self.root.destroy()

    def on_key_release(self, ev):
        k = ev.keysym
        if k == "Up": self.key_throttle = False
        elif k == "space": self.key_brake = False

    def reset(self):
        self.v_ms = 0.0; self.gear = 1; self.throttle = 0.0
        self.braking = 0.0; self.shift_timer = 0.0
        self.odometer_km = 0.0; self.want_throttle = IDLE_THROTTLE


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

        current_ratio = GEAR_RATIOS[self.gear-1]
        rpm = speed_to_rpm(self.v_ms, current_ratio)

        if self.shift_timer > 0.0:
            self.shift_timer -= dt
        if self.shift_timer <= 0.0 and self.gear < len(GEAR_RATIOS):
            if rpm > SHIFT_UP_RPM and self.throttle > 0.25:
                self.gear += 1; self.shift_timer = SHIFT_COOLDOWN
                current_ratio = GEAR_RATIOS[self.gear-1]; rpm = speed_to_rpm(self.v_ms, current_ratio)
        if self.shift_timer <= 0.0 and self.gear > 1:
            if rpm < SHIFT_DOWN_RPM and (self.throttle > 0.35 or self.v_ms < 6.0):
                self.gear -= 1; self.shift_timer = SHIFT_COOLDOWN
                current_ratio = GEAR_RATIOS[self.gear-1]; rpm = speed_to_rpm(self.v_ms, current_ratio)


        engine_torque = interp_curve(rpm, TORQUE_CURVE) * self.throttle
        if self.gear == 1 and self.braking == 0.0 and self.throttle <= IDLE_THROTTLE + 0.02:
            engine_torque += CREEP_TORQUE
        if self.throttle <= IDLE_THROTTLE + 0.01:
            engine_torque -= (ENGINE_BRAKE_K * max(0.0, rpm - IDLE_RPM) / 1000.0)

        if self.shift_timer > 0.0:
            drive_force = 0.0
        else:
            wheel_torque = engine_torque * current_ratio * FINAL_DRIVE * DRIVETRAIN_EFF
            drive_force = clamp(wheel_torque / WHEEL_RADIUS_M, -MU_TIRE*MASS_KG*9.81, MU_TIRE*MASS_KG*9.81)

        F_drag = drag_force(self.v_ms)
        F_roll = rolling_resistance()
        brake_force = clamp(MASS_KG*MAX_BRAKE_DECEL*self.braking, 0.0, MU_TIRE*MASS_KG*9.81) if self.braking>0 else 0.0

        F_net = drive_force - F_drag - F_roll - brake_force
        a = F_net / MASS_KG
        self.v_ms = max(0.0, self.v_ms + a*dt)
        if self.v_ms < 0.02 and self.throttle <= IDLE_THROTTLE + 0.01 and self.braking == 0.0:
            self.v_ms = 0.0
        self.odometer_km += (self.v_ms * dt) / 1000.0

   
        G = self.geom; cx, cy, r = G["cx"], G["cy"], G["r"]
        kmh = self.v_ms * 3.6
        t = clamp(kmh / SPEED_MAX_KMH, 0.0, 1.0)
        ang = math.radians(lerp(self.start_deg, self.end_deg, t))
        nx = cx + math.cos(ang) * (r - r*0.16)
        ny = cy - math.sin(ang) * (r - r*0.16)
        self.canvas.coords(self.items["needle"], cx, cy, nx, ny)

        self.canvas.itemconfigure(self.items["speed_text"], text=f"{int(round(kmh)):3d} km/h")
        self.canvas.itemconfigure(self.items["txt_gear"], text=f"Gear: {self.gear} / {len(GEAR_RATIOS)}")
        self.canvas.itemconfigure(self.items["txt_rpm"],  text=f"RPM : {int(rpm):,}")

        bar_left, bar_right = G["bar_left"], G["bar_right"]
        th_y, br_y, bar_h = G["th_y"], G["br_y"], G["bar_h"]
        th_px = bar_left + (bar_right - bar_left) * clamp(self.throttle, 0.0, 1.0)
        br_px = bar_left + (bar_right - bar_left) * clamp(self.braking, 0.0, 1.0)
        self.canvas.coords(self.items["th_fg"], bar_left, th_y, th_px, th_y + bar_h)
        self.canvas.coords(self.items["br_fg"], bar_left, br_y, br_px, br_y + bar_h)

        self.canvas.itemconfigure(self.items["txt_driveF"], text=f"Drive F: {int(drive_force):>6d} N")
        self.canvas.itemconfigure(self.items["txt_dragF"],  text=f"Drag  F: {int(F_drag):>6d} N")
        self.canvas.itemconfigure(self.items["txt_rollF"],  text=f"Roll  F: {int(F_roll):>6d} N")
        self.canvas.itemconfigure(self.items["txt_brakeF"], text=f"Brake F: {int(brake_force):>6d} N")
        self.canvas.itemconfigure(self.items["txt_accel"],  text=f"Accel  : {a:6.2f} m/s²")
        self.canvas.itemconfigure(self.items["txt_odo"],    text=f"Odo    : {self.odometer_km:7.2f} km")

        self.root.after(int(1000/UPDATE_HZ), self.update_loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = CarSimApp(root)
    place_bottom_right(root, width=640, height=360, margin=20)
    root.mainloop()












