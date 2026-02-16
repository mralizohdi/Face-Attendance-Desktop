# -*- coding: utf-8 -*-
import time
import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont
from PIL import Image, ImageTk
from datetime import datetime, timedelta

import numpy as np
import cv2

import face_db as db
import cv_engine as eng
import config_store as cfgs


def pick_font(root):
    """Prefer Persian-friendly fonts if available."""
    try:
        fam = set(tkfont.families(root))
    except Exception:
        fam = set()
    if "B Nazanin" in fam:
        return "B Nazanin"
    if "Tahoma" in fam:
        return "Tahoma"
    return "Segoe UI"

# -------- Settings --------
ADMIN_PASSWORD = "987887787"
MIN_ENROLL_SAMPLES = 5

# Colors
C_TEXT = "#1f2937"
C_GREEN = "#2e7d32"
C_RED = "#c62828"
C_BG_IDLE = "#ffebee"
C_BG_ACTIVE = "#e8f5e9"
C_PANEL = "#ffffff"
C_BORDER = "#e5e7eb"
C_NAV_BG = "#ffffff"
C_NAV_ACTIVE = "#1976d2"

def _s(x):
    return "" if x is None else str(x).strip()

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Face Attendance System (Instructor Zahdi)")
        self.root.configure(bg="white")
        self.root.geometry("1200x760")
        self.root.minsize(980, 640)

        try:
            ttk.Style().theme_use("clam")
        except Exception:
            pass

        self.cfg = cfgs.load_config()
        self.classes = list(self.cfg.get("classes", [self.cfg.get("default_class_name","OS_Lab")]))

        # Runtime vars (attendance settings controlled in Admin)
        self.sim_th = tk.DoubleVar(value=float(self.cfg.get("default_similarity_threshold", 0.50)))
        self.score_th_att = tk.DoubleVar(value=float(self.cfg.get("default_face_score_threshold", 0.90)))
        self.cooldown_h = tk.DoubleVar(value=float(self.cfg.get("cooldown_hours", 12.0)))
        self.mode = None  # None | \'attendance\' | \'enroll\'
        self.current_page = "attendance"
        self.last_capture_t = 0.0
        self.enroll_samples = []
        self.is_fullscreen = False

        self.cooldown_hours = float(self.cfg.get("cooldown_hours", 12.0))
        self.last_recorded = db.build_last_records(hours=max(24.0, self.cooldown_hours + 2.0))

        # Camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Camera", "Camera not found or access denied")
        self.last_frame = None

        # CV engine cache
        self.detector = None
        self.recognizer = None
        self._detector_score_cache = None

        # ---------- Top bar (turns Red/Green) ----------
        self.top = tk.Frame(root, bg=C_RED)
        self.top.pack(fill="x")

        self.header = tk.Frame(self.top, bg=C_RED)
        self.header.pack(fill="x")

        self.title_lbl = tk.Label(
            self.header, text="Face Attendance System (Instructor Zahdi)",
            bg=C_RED, fg="white", font=(pick_font(self.root), 16, "bold"), anchor="w", justify="left"
        )
        self.title_lbl.pack(side="left", padx=14, pady=12)

        self.status_lbl = tk.Label(
            self.header, text="●  STOP",
            bg=C_RED, fg="white", font=(pick_font(self.root), 12, "bold"),
            padx=14, pady=6
        )
        self.status_lbl.pack(side="right", padx=14, pady=12)

        self.btnbar = tk.Frame(self.top, bg=C_RED)
        self.btnbar.pack(fill="x")
        self.full_btn = tk.Button(
            self.btnbar, text="Fullscreen", command=self.toggle_fullscreen,
            bg="#111827", fg="white", relief="flat", padx=12, pady=6
        )
        self.full_btn.pack(side="left", padx=14, pady=(0,10))

        self.root.bind("<Escape>", lambda e: self.request_exit_fullscreen())

        # ---------- Main area: pages (left) + main menu (right) ----------
        self.main = tk.Frame(root, bg="white")
        self.main.pack(fill="both", expand=True)

        self.pages = tk.Frame(self.main, bg="white")
        self.pages.pack(side="left", fill="both", expand=True)

        self.nav = tk.Frame(self.main, bg=C_NAV_BG, highlightbackground=C_BORDER, highlightthickness=1)
        self.nav.pack(side="right", fill="y")

        tk.Label(self.nav, text="Menu", bg=C_NAV_BG, fg=C_TEXT, font=(pick_font(self.root), 12, "bold"))\
            .pack(anchor="e", padx=12, pady=(12,8))

        self.btn_nav_att = tk.Button(self.nav, text="Attendance", command=lambda: self.show_page("attendance"),
                                     bg=C_NAV_ACTIVE, fg="white", relief="flat", padx=14, pady=10)
        self.btn_nav_att.pack(fill="x", padx=12, pady=6)

        self.btn_nav_enroll = tk.Button(self.nav, text="Enrollment", command=lambda: self.show_page("enroll"),
                                        bg="#334155", fg="white", relief="flat", padx=14, pady=10)
        self.btn_nav_enroll.pack(fill="x", padx=12, pady=6)

        self.btn_nav_admin = tk.Button(self.nav, text="Admin", command=lambda: self.show_page("admin"),
                                       bg="#334155", fg="white", relief="flat", padx=14, pady=10)
        self.btn_nav_admin.pack(fill="x", padx=12, pady=6)

        # Pages
        self.page_att = tk.Frame(self.pages, bg="white")
        self.page_enroll = tk.Frame(self.pages, bg="white")
        self.page_admin = tk.Frame(self.pages, bg="white")

        for p in (self.page_att, self.page_enroll, self.page_admin):
            p.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._build_attendance(self.page_att)
        self._build_enroll(self.page_enroll)
        self._build_admin(self.page_admin)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Default page: Attendance
        self.show_page("attendance")
        self._update_mode_ui()
        self._tick()

    # ---------- Navigation ----------
    def show_page(self, name: str):
        self.current_page = name
        if name == "attendance":
            self.page_att.lift()
            self._set_nav_active(self.btn_nav_att)
        elif name == "enroll":
            self.page_enroll.lift()
            self._set_nav_active(self.btn_nav_enroll)
        else:
            self.page_admin.lift()
            self._set_nav_active(self.btn_nav_admin)

    def _set_nav_active(self, active_btn):
        for b in (self.btn_nav_att, self.btn_nav_enroll, self.btn_nav_admin):
            if b is active_btn:
                b.configure(bg=C_NAV_ACTIVE)
            else:
                b.configure(bg="#334155")

    # ---------- Engine ----------
    def ensure_engine(self, face_score_th: float):
        face_score_th = float(face_score_th)
        if self.detector is None or self._detector_score_cache is None or abs(self._detector_score_cache - face_score_th) > 1e-6:
            self.detector = eng.make_detector(score_thresh=face_score_th)
            self._detector_score_cache = face_score_th
        if self.recognizer is None:
            self.recognizer = eng.make_recognizer()

    # ---------- Layout helpers ----------
    def _make_split(self, parent):
        outer = tk.Frame(parent, bg="white")
        outer.pack(fill="both", expand=True)

        video_wrap = tk.Frame(outer, bg=C_BG_IDLE, highlightbackground=C_BORDER, highlightthickness=1)
        video_wrap.pack(side="left", fill="both", expand=True, padx=14, pady=14)

        controls = tk.Frame(outer, bg=C_PANEL, highlightbackground=C_BORDER, highlightthickness=1)
        controls.pack(side="right", fill="y", padx=(0,14), pady=14)

        video_inner = tk.Frame(video_wrap, bg=video_wrap["bg"])
        video_inner.pack(fill="both", expand=True, padx=10, pady=10)
        video_inner.rowconfigure(0, weight=1)
        video_inner.columnconfigure(0, weight=1)
        return controls, video_wrap, video_inner

    def _banner(self, parent, text="", kind="info"):
        colors = {
            "info": ("#e3f2fd", "#0d47a1"),
            "ok": ("#e8f5e9", "#1b5e20"),
            "warn": ("#fff8e1", "#e65100"),
            "err": ("#ffebee", "#b71c1c"),
        }
        bg, fg = colors.get(kind, colors["info"])
        fr = tk.Frame(parent, bg=bg, highlightbackground=C_BORDER, highlightthickness=1)
        # tk.Message auto-wraps inside the box (prevents text from going outside the box)
        msg = tk.Message(
            fr, text=text, bg=bg, fg=fg,
            font=(pick_font(self.root), 11, "bold"),
            justify="left", anchor="e", width=360
        )
        msg.pack(fill="x", padx=10, pady=8)
        return fr, msg

    def _fmt_att_message(self, name: str, student_id: str, cls: str, is_duplicate: bool, date_str: str | None = None):
        # Format requested: each field on its own line (no "|").
        # 1) student id  2) name  3) class  4) status
        status = "Already recorded (cooldown)" if is_duplicate else "Attendance recorded"
        return f"{student_id}\n{name}\n{cls}\n{status}"


    # ---------- Attendance page ----------
    def _build_attendance(self, parent):
        controls, self.video_wrap_att, video_inner = self._make_split(parent)

        tk.Label(controls, text="Attendance (Auto)", bg=C_PANEL, fg=C_TEXT, font=(pick_font(self.root), 12, "bold"),
                 anchor="w", justify="left")\
            .pack(fill="x", padx=12, pady=(12,6))
        tk.Label(controls, text="Each student can be recorded once every 24 hours.", bg=C_PANEL, fg=C_TEXT,
                 anchor="w", justify="left")\
            .pack(fill="x", padx=12, pady=(0,10))

        # Attendance settings are configured in the Admin panel (removed here to keep UI simple).
        self.btn_att_toggle = tk.Button(
            controls, text="Start Attendance", command=self.toggle_attendance,
            bg=C_GREEN, fg="white", relief="flat",
            font=(pick_font(self.root), 13, "bold"),
            padx=12, pady=14
        )
        self.btn_att_toggle.pack(fill="x", padx=12, pady=(8,10))
        # Recognition result panel (one field per line)
        self.att_info_box = tk.Frame(controls, bg=C_PANEL, highlightbackground=C_BORDER, highlightthickness=1)
        self.att_info_box.pack(fill="x", padx=12, pady=(6,12))

        fnt = (pick_font(self.root), 11)
        fnt_b = (pick_font(self.root), 11, "bold")

        self.att_id_var = tk.StringVar(value="Student ID: —")
        self.att_name_var = tk.StringVar(value="Name: —")
        self.att_class_var = tk.StringVar(value="Class: —")
        self.att_status_var = tk.StringVar(value="Status: Ready")

        # Rows
        self.att_id_lbl = tk.Label(self.att_info_box, textvariable=self.att_id_var, bg="white", fg=C_TEXT,
                                   anchor="w", justify="left", wraplength=360, font=fnt_b, padx=10, pady=6)
        self.att_id_lbl.pack(fill="x", padx=8, pady=(8,4))

        self.att_name_lbl = tk.Label(self.att_info_box, textvariable=self.att_name_var, bg="white", fg=C_TEXT,
                                     anchor="w", justify="left", wraplength=360, font=fnt, padx=10, pady=6)
        self.att_name_lbl.pack(fill="x", padx=8, pady=4)

        self.att_class_lbl = tk.Label(self.att_info_box, textvariable=self.att_class_var, bg="white", fg=C_TEXT,
                                      anchor="w", justify="left", wraplength=360, font=fnt, padx=10, pady=6)
        self.att_class_lbl.pack(fill="x", padx=8, pady=4)

        # Status (colored)
        self.att_status_lbl = tk.Label(self.att_info_box, textvariable=self.att_status_var, bg="#e3f2fd", fg="#0d47a1",
                                       anchor="w", justify="left", wraplength=360, font=fnt_b, padx=10, pady=8)
        self.att_status_lbl.pack(fill="x", padx=8, pady=(4,8))

        self.video_label_att = tk.Label(video_inner, bg=video_inner["bg"])
        self.video_label_att.grid(row=0, column=0, sticky="nsew")

    # ---------- Enroll page ----------
    def _build_enroll(self, parent):
        controls, self.video_wrap_enroll, video_inner = self._make_split(parent)

        tk.Label(controls, text="Enrollment", bg=C_PANEL, fg=C_TEXT, font=(pick_font(self.root), 12, "bold"),
                 anchor="w", justify="left").pack(anchor="e", padx=12, pady=(12,6))

        tk.Label(controls, text="Student ID", bg=C_PANEL, fg=C_TEXT, anchor="w", justify="left").pack(fill="x", padx=12)
        self.enroll_id = tk.StringVar()
        ttk.Entry(controls, textvariable=self.enroll_id, width=30, justify="left").pack(fill="x", padx=12, pady=(0,10))

        tk.Label(controls, text="Full Name", bg=C_PANEL, fg=C_TEXT, anchor="w", justify="left").pack(fill="x", padx=12)
        self.enroll_name = tk.StringVar()
        ttk.Entry(controls, textvariable=self.enroll_name, width=30, justify="left").pack(fill="x", padx=12, pady=(0,10))

        tk.Label(controls, text="Class", bg=C_PANEL, fg=C_TEXT, anchor="w", justify="left").pack(fill="x", padx=12)
        self.enroll_class = tk.StringVar(value=self.cfg.get("default_class_name","OS_Lab"))
        self.enroll_class_combo = ttk.Combobox(controls, textvariable=self.enroll_class, values=self.classes, state="readonly", width=26)
        self.enroll_class_combo.pack(fill="x", padx=12, pady=(0,10))

        tk.Label(controls, text="Target samples (default 10, min 5)", bg=C_PANEL, fg=C_TEXT, anchor="w", justify="left").pack(fill="x", padx=12)
        self.enroll_target = tk.IntVar(value=int(self.cfg.get("enroll_samples_target", 10)))
        ttk.Spinbox(controls, from_=MIN_ENROLL_SAMPLES, to=50, textvariable=self.enroll_target, width=10, justify="left")\
            .pack(fill="x", padx=12, pady=(0,10))

        tk.Label(controls, text="Capture interval (sec) (default 2)", bg=C_PANEL, fg=C_TEXT, anchor="w", justify="left").pack(fill="x", padx=12)
        self.enroll_interval = tk.DoubleVar(value=2.0)
        ttk.Spinbox(controls, from_=0.8, to=10.0, increment=0.2, textvariable=self.enroll_interval, width=10, justify="left")\
            .pack(fill="x", padx=12, pady=(0,10))

        tk.Label(controls, text="Face Score Threshold (quality)", bg=C_PANEL, fg=C_TEXT, anchor="w", justify="left").pack(fill="x", padx=12)
        self.score_th_enroll = tk.DoubleVar(value=float(self.cfg.get("default_face_score_threshold", 0.90)))
        ttk.Spinbox(controls, from_=0.5, to=0.99, increment=0.01, textvariable=self.score_th_enroll, width=10, justify="left")\
            .pack(fill="x", padx=12, pady=(0,10))

        self.enroll_progress = ttk.Progressbar(controls, maximum=100, value=0)
        self.enroll_progress.pack(fill="x", padx=12, pady=(6,6))

        self.btn_enroll_toggle = tk.Button(
            controls, text="Start Enrollment", command=self.toggle_enroll,
            bg=C_GREEN, fg="white", relief="flat",
            font=(pick_font(self.root), 13, "bold"),
            padx=12, pady=14
        )
        self.btn_enroll_toggle.pack(fill="x", padx=12, pady=(8,8))

        self.enroll_banner, self.enroll_banner_lbl = self._banner(controls, "Ready", "info")
        self.enroll_banner.pack(fill="x", padx=12, pady=(6,12))

        self.video_label_enroll = tk.Label(video_inner, bg=video_inner["bg"])
        self.video_label_enroll.grid(row=0, column=0, sticky="nsew")

    # ---------- Admin page ----------
    def _build_admin(self, parent):
        self.admin_logged_in = False

        outer = tk.Frame(parent, bg="white")
        outer.pack(fill="both", expand=True, padx=14, pady=14)

        left = tk.Frame(outer, bg=C_PANEL, highlightbackground=C_BORDER, highlightthickness=1)
        left.pack(side="right", fill="y", padx=(14,0))

        right = tk.Frame(outer, bg=C_PANEL, highlightbackground=C_BORDER, highlightthickness=1)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="Admin", bg=C_PANEL, fg=C_TEXT, font=(pick_font(self.root), 12, "bold"))\
            .pack(anchor="w", padx=12, pady=(12,6))

        tk.Label(left, text="Admin Password", bg=C_PANEL, fg=C_TEXT).pack(anchor="w", padx=12)
        self.admin_pw = tk.StringVar()
        self.admin_pw_entry = ttk.Entry(left, textvariable=self.admin_pw, show="*", width=22)
        self.admin_pw_entry.pack(anchor="w", padx=12, pady=(0,8))
        self.admin_pw_entry.bind('<Return>', self._on_admin_enter)

        self.admin_toggle_btn = tk.Button(
            left, text="Login", command=self.admin_toggle,
            bg="#111827", fg="white", relief="flat", padx=10, pady=10
        )
        self.admin_toggle_btn.pack(fill="x", padx=12, pady=(0,10))

        self.admin_status = tk.StringVar(value="")
        tk.Label(left, textvariable=self.admin_status, bg=C_PANEL, fg="#0f172a")\
            .pack(anchor="w", padx=12, pady=(0,10))

        self.admin_frame = tk.Frame(right, bg=C_PANEL)
        self.admin_frame.pack(fill="both", expand=True)
        self.admin_frame.pack_forget()

        sec = tk.Frame(self.admin_frame, bg=C_PANEL)
        sec.pack(fill="both", expand=True, padx=12, pady=12)

        col1 = tk.Frame(sec, bg=C_PANEL)
        col1.pack(side="left", fill="both", expand=True)

        top = tk.Frame(col1, bg=C_PANEL)
        top.pack(fill="x", pady=(0,8))
        tk.Label(top, text="Students", bg=C_PANEL, fg=C_TEXT, font=(pick_font(self.root), 12, "bold"))\
            .pack(side="left")

        btns = tk.Frame(col1, bg=C_PANEL)
        btns.pack(fill="x", pady=(0,8))
        tk.Button(btns, text="Refresh", command=self.refresh_students, bg="#334155", fg="white",
                  relief="flat", padx=10, pady=6).pack(side="left", padx=(0,6))
        tk.Button(btns, text="Delete Selected", command=self.delete_selected, bg=C_RED, fg="white",
                  relief="flat", padx=10, pady=6).pack(side="left")

        self.students_lb = tk.Listbox(col1, height=18, font=("Consolas", 12))
        self.students_lb.pack(fill="both", expand=True)

        col2 = tk.Frame(sec, bg=C_PANEL, width=420)
        col2.pack(side="right", fill="y", padx=(14,0))
        col2.pack_propagate(False)

        cls_box = tk.Frame(col2, bg=C_PANEL, highlightbackground=C_BORDER, highlightthickness=1)
        cls_box.pack(fill="x", pady=(0,12))
        tk.Label(cls_box, text="Classes", bg=C_PANEL, fg=C_TEXT, font=(pick_font(self.root), 12, "bold"))\
            .pack(anchor="w", padx=10, pady=(10,6))
        self.class_list = tk.Listbox(cls_box, height=6)
        self.class_list.pack(fill="x", padx=10)
        row = tk.Frame(cls_box, bg=C_PANEL)
        row.pack(fill="x", padx=10, pady=10)
        self.new_class = tk.StringVar()
        ttk.Entry(row, textvariable=self.new_class, width=18).pack(side="left", padx=(0,8))
        tk.Button(row, text="Add", command=self.add_class, bg=C_GREEN, fg="white",
                  relief="flat", padx=10, pady=6).pack(side="left", padx=(0,8))
        tk.Button(row, text="Delete", command=self.delete_class, bg=C_RED, fg="white",
                  relief="flat", padx=10, pady=6).pack(side="left")

        st_box = tk.Frame(col2, bg=C_PANEL, highlightbackground=C_BORDER, highlightthickness=1)
        st_box.pack(fill="x")
        tk.Label(st_box, text="Settings", bg=C_PANEL, fg=C_TEXT, font=(pick_font(self.root), 12, "bold"))\
            .pack(anchor="w", padx=10, pady=(10,6))

        form = tk.Frame(st_box, bg=C_PANEL)
        form.pack(fill="x", padx=10, pady=10)

        self.cfg_default_class = tk.StringVar(value=self.cfg.get("default_class_name","OS_Lab"))
        self.cfg_sim = tk.DoubleVar(value=float(self.cfg.get("default_similarity_threshold",0.50)))
        self.cfg_score = tk.DoubleVar(value=float(self.cfg.get("default_face_score_threshold",0.90)))
        self.cfg_interval = tk.DoubleVar(value=float(self.cfg.get("capture_interval_sec",2.0)))
        self.cfg_enroll_n = tk.IntVar(value=int(self.cfg.get("enroll_samples_target",10)))
        self.cfg_cool_h = tk.DoubleVar(value=float(self.cfg.get("cooldown_hours",24.0)))

        # Grid layout (avoid diagonal / messy alignment)
        form.grid_columnconfigure(0, weight=1)
        form.grid_columnconfigure(1, weight=0)

        def grid_row(r, label_text, widget):
            tk.Label(form, text=label_text, bg=C_PANEL, fg=C_TEXT,
                     anchor="w", justify="left", width=18).grid(row=r, column=1, sticky="e", padx=(6,8), pady=6)
            widget.grid(row=r, column=0, sticky="ew", padx=(8,6), pady=6)

        self.cfg_class_combo = ttk.Combobox(form, textvariable=self.cfg_default_class, values=self.classes, state="readonly")
        grid_row(0, "Default Class", self.cfg_class_combo)

        sim_spin = ttk.Spinbox(form, from_=0.2, to=0.8, increment=0.01, textvariable=self.cfg_sim, width=9, justify="left")
        grid_row(1, "Similarity Threshold", sim_spin)

        score_spin = ttk.Spinbox(form, from_=0.5, to=0.99, increment=0.01, textvariable=self.cfg_score, width=9, justify="left")
        grid_row(2, "Face Score", score_spin)

        interval_spin = ttk.Spinbox(form, from_=0.5, to=5.0, increment=0.5, textvariable=self.cfg_interval, width=9, justify="left")
        grid_row(3, "Capture interval (sec)", interval_spin)

        enroll_spin = ttk.Spinbox(form, from_=5, to=50, increment=1, textvariable=self.cfg_enroll_n, width=9, justify="left")
        grid_row(4, "Enrollment samples", enroll_spin)

        cooldown_spin = ttk.Spinbox(form, from_=1, to=48, increment=1, textvariable=self.cfg_cool_h, width=9, justify="left")
        grid_row(5, "Cooldown (hours)", cooldown_spin)

        tk.Button(st_box, text="Save", command=self.save_settings, bg="#1976d2", fg="white",
                  relief="flat", padx=12, pady=8)\
            .pack(anchor="e", padx=10, pady=(0,10))
    def _on_admin_enter(self, event=None):
        # Enter key should only attempt login, not logout.
        if not self.admin_logged_in:
            self.admin_toggle()
        return "break"



    def admin_toggle(self):
        if not self.admin_logged_in:
            if _s(self.admin_pw.get()) == ADMIN_PASSWORD:
                self.admin_logged_in = True
                self.admin_toggle_btn.configure(text="Logout", bg=C_RED)
                self.admin_frame.pack(fill="both", expand=True)
                self.refresh_students()
                self.refresh_classes_ui()
            else:
                messagebox.showerror("Admin", "Wrong password")
                self.admin_frame.pack_forget()
        else:
            self.admin_logged_in = False
            self.admin_pw.set("")
            self.admin_toggle_btn.configure(text="Login", bg="#111827")
            self.admin_frame.pack_forget()

    def refresh_students(self):
        self.students_lb.delete(0, tk.END)
        meta = db.load_meta()
        for sid, info in sorted(meta.items(), key=lambda x: x[0]):
            n = db.load_features(sid).shape[0]
            name = info.get("name","")
            cls = info.get("class","")
            self.students_lb.insert(tk.END, f"{sid} | {name} | Class={cls} | samples={n}")

    def delete_selected(self):
        sel = self.students_lb.curselection()
        if not sel:
            return
        item = self.students_lb.get(sel[0])
        sid = item.split("|")[0].strip()
        if messagebox.askyesno("Delete", f"Delete permanently?\n{sid}"):
            db.delete_student(sid, delete_logs=True)
            self.refresh_students()

    def refresh_classes_ui(self):
        self.class_list.delete(0, tk.END)
        for c in self.classes:
            self.class_list.insert(tk.END, c)

    def _sync_class_dropdowns(self):
        if not self.classes:
            self.classes = [self.cfg.get("default_class_name","OS_Lab")]
        self.enroll_class_combo.configure(values=self.classes)
        self.cfg_class_combo.configure(values=self.classes)
        if self.enroll_class.get() not in self.classes:
            self.enroll_class.set(self.classes[0])
        if self.cfg_default_class.get() not in self.classes:
            self.cfg_default_class.set(self.classes[0])

    def add_class(self):
        if not self.admin_logged_in:
            return
        name = _s(self.new_class.get())
        if not name or name in self.classes:
            return
        self.classes.append(name)
        self.classes = list(dict.fromkeys(self.classes))
        self.new_class.set("")
        self.refresh_classes_ui()
        self._sync_class_dropdowns()

    def delete_class(self):
        if not self.admin_logged_in:
            return
        sel = self.class_list.curselection()
        if not sel:
            return
        name = self.class_list.get(sel[0])
        if len(self.classes) <= 1:
            return
        if messagebox.askyesno("Class", f"Delete this class?\n{name}"):
            self.classes = [c for c in self.classes if c != name]
            self.refresh_classes_ui()
            self._sync_class_dropdowns()

    def save_settings(self):
        if not self.admin_logged_in:
            messagebox.showwarning("Admin", "Please login first")
            return

        default_class = self.cfg_default_class.get()
        if default_class not in self.classes:
            self.classes.insert(0, default_class)

        cfgs.save_config({
            "default_class_name": default_class,
            "classes": self.classes,
            "default_similarity_threshold": float(self.cfg_sim.get()),
            "default_face_score_threshold": float(self.cfg_score.get()),
            "capture_interval_sec": float(self.cfg_interval.get()),
            "enroll_samples_target": int(self.cfg_enroll_n.get()),
            "cooldown_hours": float(self.cfg_cool_h.get()),
        })
        self.cfg = cfgs.load_config()
        self.classes = list(self.cfg.get("classes", self.classes))

        # Runtime vars (attendance settings controlled in Admin)
        self.sim_th = tk.DoubleVar(value=float(self.cfg.get("default_similarity_threshold", 0.50)))
        self.score_th_att = tk.DoubleVar(value=float(self.cfg.get("default_face_score_threshold", 0.90)))
        self.cooldown_h = tk.DoubleVar(value=float(self.cfg.get("cooldown_hours", 12.0)))

        self._sync_class_dropdowns()

        self.sim_th.set(float(self.cfg.get("default_similarity_threshold",0.50)))
        self.score_th_att.set(float(self.cfg.get("default_face_score_threshold",0.90)))
        self.score_th_enroll.set(float(self.cfg.get("default_face_score_threshold",0.90)))
        self.enroll_interval.set(2.0)
        self.enroll_target.set(max(MIN_ENROLL_SAMPLES, int(self.cfg.get("enroll_samples_target",10))))
        self.cooldown_h.set(float(self.cfg.get("cooldown_hours",12.0)))
        self.cooldown_hours = float(self.cfg.get("cooldown_hours",12.0))

        self.detector = None
        self._detector_score_cache = None
        messagebox.showinfo("Admin", "Saved")

    # ---------- UI state ----------
    def _update_mode_ui(self):
        active = self.mode in ("enroll", "attendance")
        bar = C_GREEN if active else C_RED
        bg = C_BG_ACTIVE if active else C_BG_IDLE

        if self.mode == "attendance":
            text = "●  REC (Attendance)"
        elif self.mode == "enroll":
            text = "●  REC (Enrollment)"
        else:
            text = "●  STOP"

        self.top.configure(bg=bar)
        self.header.configure(bg=bar)
        self.btnbar.configure(bg=bar)
        self.title_lbl.configure(bg=bar)
        self.status_lbl.configure(bg=bar, text=text)

        for wrap in [getattr(self, "video_wrap_att", None), getattr(self, "video_wrap_enroll", None)]:
            if wrap is not None:
                wrap.configure(bg=bg)
        if hasattr(self, "video_label_att"):
            self.video_label_att.configure(bg=bg)
        if hasattr(self, "video_label_enroll"):
            self.video_label_enroll.configure(bg=bg)

        if self.mode == "attendance":
            self.btn_att_toggle.configure(text="Stop Attendance", bg=C_RED)
        else:
            self.btn_att_toggle.configure(text="Start Attendance", bg=C_GREEN)

        if self.mode == "enroll":
            self.btn_enroll_toggle.configure(text="Stop Enrollment", bg=C_RED)
        else:
            self.btn_enroll_toggle.configure(text="Start Enrollment", bg=C_GREEN)

    def _set_att_banner(self, text, kind, sid="—", name="—", cls="—"):
        """
        Update the Attendance info panel (4 separate lines).
        kind: info / ok / warn / err (controls status color)
        """
        if not hasattr(self, "att_id_var"):
            return

        self.att_id_var.set(f"Student ID: {sid}")
        self.att_name_var.set(f"Name: {name}")
        self.att_class_var.set(f"Class: {cls}")
        self.att_status_var.set(f"Status: {text}")

        colors = {
            "info": ("#e3f2fd", "#0d47a1"),
            "ok": ("#e8f5e9", "#1b5e20"),
            "warn": ("#fff8e1", "#e65100"),
            "err": ("#ffebee", "#b71c1c"),
        }
        bg, fg = colors.get(kind, colors["info"])
        if hasattr(self, "att_status_lbl"):
            self.att_status_lbl.configure(bg=bg, fg=fg)

    def _set_enroll_banner(self, text, kind):
        colors = {"info": ("#e3f2fd","#0d47a1"), "ok": ("#e8f5e9","#1b5e20"),
                  "warn": ("#fff8e1","#e65100"), "err": ("#ffebee","#b71c1c")}
        bg, fg = colors.get(kind, colors["info"])
        self.enroll_banner.configure(bg=bg)
        self.enroll_banner_lbl.configure(bg=bg, fg=fg, text=text)

    # ---------- Actions ----------
    def toggle_attendance(self):
        if self.mode == "attendance":
            self.mode = None
            self._set_att_banner("Stopped", "warn", sid="—", name="—", cls="—")
        else:
            if not db.load_meta():
                self._set_att_banner("No enrolled students yet", "err", sid="—", name="—", cls="—")
                return
            self.mode = "attendance"
            self.last_capture_t = 0.0
            self._set_att_banner("Attendance started. Students may walk in front of the camera.", "info", sid="—", name="—", cls="—")
        self._update_mode_ui()

    def toggle_enroll(self):
        if self.mode == "enroll":
            if len(self.enroll_samples) >= MIN_ENROLL_SAMPLES:
                self._finalize_enroll()
                self._set_enroll_banner("Enrollment saved", "ok")
            else:
                self._set_enroll_banner(f"Not enough samples (minimum {MIN_ENROLL_SAMPLES})", "warn")
            self.mode = None
            self._update_mode_ui()
            return

        sid = _s(self.enroll_id.get())
        name = _s(self.enroll_name.get())

        if not sid:
            self._set_enroll_banner("Enter Student ID", "err")
            return
        if not name:
            self._set_enroll_banner("Enter Full Name", "err")
            return

        meta = db.load_meta()
        if sid in meta:
            self._set_enroll_banner("This Student ID is already enrolled", "warn")
            return

        cls = self.enroll_class.get() or self.cfg.get("default_class_name","OS_Lab")
        if cls not in self.classes:
            self._set_enroll_banner("Invalid class. Please configure it in Admin.", "err")
            return

        self.mode = "enroll"
        self.enroll_samples = []
        self.last_capture_t = 0.0
        self.enroll_progress.configure(value=0)
        self._set_enroll_banner("Enrollment started. Please face the camera.", "info")
        self._update_mode_ui()

    # ---------- Matching ----------
    def best_match(self, feat: np.ndarray, threshold: float):
        meta = db.load_meta()
        best_id = None
        best_sim = -1.0
        for sid in meta.keys():
            feats = db.load_features(sid)
            if feats.shape[0] == 0:
                continue
            sims = [eng.cosine_sim(self.recognizer, feat, f) for f in feats]
            s = float(np.max(sims))
            if s > best_sim:
                best_sim = s
                best_id = sid
        if best_sim >= threshold:
            return best_id, best_sim
        return None, best_sim

    def _process_frame(self, bgr):
        face_score = float(self.score_th_enroll.get()) if self.mode == "enroll" else float(self.score_th_att.get())
        self.ensure_engine(face_score)

        # --- Feature extraction ---
        if self.mode == "attendance":
            faces_mat = eng.detect_faces(self.detector, bgr)
            if faces_mat is None or len(faces_mat) == 0:
                return bgr, [], "NO_FACE"

            # filter by detector score (index 4 is confidence)
            faces = []
            for row in faces_mat:
                try:
                    if float(row[4]) >= face_score:
                        faces.append(row)
                except Exception:
                    faces.append(row)

            if len(faces) == 0:
                faces = list(faces_mat)

            # prefer largest faces, cap to keep CPU reasonable
            faces = sorted(faces, key=lambda r: float(r[2] * r[3]), reverse=True)[:5]

            feats = []
            disp = bgr.copy()
            for row in faces:
                aligned = self.recognizer.alignCrop(bgr, row)
                feat = self.recognizer.feature(aligned)
                feat = np.asarray(feat, dtype=np.float32).reshape(-1)
                feats.append(feat)
                disp = eng.draw_face_box(disp, row, color=(0, 255, 0))
            code = "OK"
        else:
            feat, face_row, code = eng.extract_feature(self.detector, self.recognizer, bgr)
            disp = eng.draw_face_box(bgr, face_row, color=(0,255,0) if code=="OK" else (0,0,255))
            if code != "OK":
                return disp, None, code
            feats = feat

        # --- Rate limit captures ---
        now = time.time()
        interval = float(self.enroll_interval.get()) if self.mode=="enroll" else float(self.cfg.get("capture_interval_sec", 2.0))
        if now - self.last_capture_t < interval:
            return disp, feats, "OK_WAIT"

        self.last_capture_t = now
        return disp, feats, "OK_CAPTURE"""

    def _finalize_enroll(self):
        sid = _s(self.enroll_id.get())
        name = _s(self.enroll_name.get())
        cls = self.enroll_class.get() or self.cfg.get("default_class_name","OS_Lab")

        meta = db.load_meta()
        if sid in meta:
            return False

        meta[sid] = {"name": name, "class": cls}
        for f in self.enroll_samples:
            db.append_feature(sid, f)
        db.save_meta(meta)
        try:
            self.refresh_students()
        except Exception:
            pass
        return True

    def _enroll_step(self, feat):
        target = max(MIN_ENROLL_SAMPLES, int(self.enroll_target.get()))
        self.enroll_samples.append(feat)
        pct = int(100 * len(self.enroll_samples) / max(1, target))
        self.enroll_progress.configure(value=min(100, pct))
        self._set_enroll_banner(f"Sample captured: {len(self.enroll_samples)}/{target}", "info")

        if len(self.enroll_samples) >= target:
            ok = self._finalize_enroll()
            if ok:
                self._set_enroll_banner("Enrollment saved", "ok")
            else:
                self._set_enroll_banner("This Student ID already exists", "warn")
            self.mode = None
            self._update_mode_ui()

    def _attendance_step(self, feat):
        th = float(self.sim_th.get())
        meta = db.load_meta()

        sid, sim = self.best_match(feat, th)
        if sid is None:
            self._set_att_banner(f"Unknown (best={sim:.3f})", "warn", sid="—", name="—", cls="—")
            return

        cls = meta.get(sid, {}).get("class", self.cfg.get("default_class_name","OS_Lab"))
        name = meta.get(sid, {}).get("name","")

        hours = float(self.cooldown_h.get())
        last = self.last_recorded.get(sid)
        if last is not None and datetime.now() - last < timedelta(hours=hours):
            self._set_att_banner("Already recorded (cooldown)", "info", sid=sid, name=name, cls=cls)
            return

        db.append_attendance_row(cls, sid, name, sim)
        self.last_recorded[sid] = datetime.now()
        self._set_att_banner("Attendance recorded", "ok", sid=sid, name=name, cls=cls)

    # ---------- Video ----------
    def _active_video_label(self):
        if getattr(self, 'current_page', 'attendance') == 'attendance':
            return self.video_label_att
        if getattr(self, 'current_page', 'attendance') == 'enroll':
            return self.video_label_enroll
        return None

    def _show_frame(self, bgr):
        lbl = self._active_video_label()
        if lbl is None:
            return

        w = max(lbl.winfo_width(), 760)
        h = max(lbl.winfo_height(), 540)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        rgb2 = cv2.resize(rgb, (max(1,nw), max(1,nh)))
        img = Image.fromarray(rgb2)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.configure(image=imgtk)

    def _tick(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame

        if self.last_frame is not None:
            bgr = self.last_frame
            if self.mode in ("enroll","attendance"):
                disp, feat, code = self._process_frame(bgr)
                if code == "OK_CAPTURE":
                    if self.mode == "enroll":
                        self._enroll_step(feat)
                    else:
                        if isinstance(feat, list):
                            for f in feat:
                                self._attendance_step(f)
                        else:
                            self._attendance_step(feat)
            else:
                disp = self.last_frame
            self._show_frame(disp)

        self.root.after(30, self._tick)

    # ---------- Fullscreen ----------
    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)

    def request_exit_fullscreen(self):
        if not self.is_fullscreen:
            return
        self._ask_password_to_exit_fullscreen()

    def _ask_password_to_exit_fullscreen(self):
        win = tk.Toplevel(self.root)
        win.title("Exit Fullscreen")
        win.transient(self.root)
        win.grab_set()
        win.configure(bg="white")
        win.geometry("360x180")
        win.resizable(False, False)

        tk.Label(win, text="Enter admin password to exit fullscreen", bg="white", fg=C_TEXT,
                 font=(pick_font(self.root), 11, "bold")).pack(pady=(18,8))
        pw = tk.StringVar()
        ent = ttk.Entry(win, textvariable=pw, show="*", width=26)
        ent.pack()
        ent.focus_set()

        msg = tk.Label(win, text="", bg="white", fg=C_RED, font=(pick_font(self.root), 10, "bold"))
        msg.pack(pady=(8,0))

        def ok():
            if _s(pw.get()) == ADMIN_PASSWORD:
                self.is_fullscreen = False
                self.root.attributes("-fullscreen", False)
                win.destroy()
            else:
                msg.configure(text="Wrong password")

        tk.Button(win, text="Confirm", command=ok, bg="#1976d2", fg="white", relief="flat", padx=10, pady=6)\
            .pack(pady=12)

    # ---------- Exit ----------
    def on_close(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()

def main():
    root = tk.Tk()
    font_name = pick_font(root)
    root.option_add('*Font', (font_name, 11))
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
