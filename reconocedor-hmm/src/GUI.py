"""Aplicación de escritorio: reconocedor de caracteres 2D/3D con HMM y guías de trazado.

Solo utiliza Tkinter y el núcleo HMM nativo (src.FeatureExtractor y src.HMMEngine).
No incluye modo ni botón de entrenamiento/calibración.

2D: dibujo directo sobre cuadrícula con trayectoria real, estimada, punto inicial y final.
3D: proyección ortográfica real de puntos sobre esfera unidad, picking (rayo-cámara ∩ esfera)
y rotación de cámara (clic izquierdo dibuja, derecho rota, rueda acerca/aleja).
"""
from __future__ import annotations

import ctypes
import json
import math
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from src.FeatureExtractor import (
    LETTERS,
    extract_2d,
    extract_3d,
    normalize_sphere,
    sphere_from_angles,
)
from src.HMMEngine import HMM, classify, project_2d, project_3d

ROOT = Path(__file__).resolve().parents[1]

# Paleta de colores

BG = "#dff1ec"              # fondo general menta suave
BG_HEADER = "#e6f4f0"       # franja superior sutil
PANEL = "#ffffff"           # superficie de tarjetas
PANEL_BORDER = "#cbe0da"    # borde fino de tarjetas
PANEL_SHADOW = "#c3dcd5"    # sombra suave de tarjetas
CANVAS_BG = "#f8fbfa"       # fondo del lienzo de dibujo
GRID_MINOR = "#e7efec"      # cuadrícula secundaria
GRID_MAJOR = "#d5e4df"      # cuadrícula principal

TEAL = "#0f7b6c"            # acento principal (botones, tabs, top-1)
TEAL_HOVER = "#149280"      # hover sobre acento principal
TEAL_LIGHT = "#e5f2ef"      # fondo suave verde azulado
NAVY = "#172536"            # trayectoria real y titulares
CYAN = "#22c3d8"            # punto inicial
CORAL = "#e6735a"           # punto final
ESTIMATED = "#8e9ca6"       # trayectoria estimada (discontinua)

BAR_SECONDARY = "#5da8e6"   # barras de probabilidad (puestos 2..27)
BAR_TRACK = "#e6eef3"       # pista de barras de progreso
TILE_BG = "#c7d7e2"         # botones de letras en barra inferior
TILE_HOVER = "#9ec0d6"      # hover de botones de letras
BTN_BG = "#ffffff"          # botones secundarios (Limpiar / Deshacer)
BTN_HOVER = "#eef6f3"       # hover botones secundarios
BTN_BORDER = "#cfe0db"      # borde botones secundarios

TEXT = "#172536"            # texto principal
MUTED = "#5e716d"           # texto secundario

#(27 letras) para la barra inferior

SPANISH_ALPHABET = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "ñ", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
]

# guías vectoriales de trazado óptimo para las 27 letras minúsculas

TRACING_GUIDES: dict[str, list[list[tuple[float, float]]]] = {
    "a": [
        [
            (0.61, 0.34), (0.55, 0.30), (0.47, 0.30), (0.38, 0.34),
            (0.30, 0.43), (0.26, 0.55), (0.28, 0.66), (0.35, 0.73),
            (0.45, 0.74), (0.55, 0.68), (0.62, 0.56), (0.64, 0.42),
            (0.61, 0.34),
        ],
        [
            (0.65, 0.30), (0.65, 0.46), (0.65, 0.63), (0.67, 0.72),
            (0.72, 0.74), (0.77, 0.67),
        ],
    ],
    "b": [
        [(0.34, 0.18), (0.34, 0.46), (0.34, 0.74)],
        [
            (0.34, 0.48), (0.42, 0.38), (0.53, 0.36), (0.63, 0.42),
            (0.67, 0.54), (0.64, 0.67), (0.54, 0.74), (0.42, 0.73),
            (0.34, 0.65),
        ],
    ],
    "c": [
        [
            (0.66, 0.40), (0.58, 0.33), (0.46, 0.33), (0.36, 0.40),
            (0.32, 0.54), (0.36, 0.67), (0.47, 0.74), (0.59, 0.73),
            (0.67, 0.66),
        ],
    ],
    "d": [
        [
            (0.65, 0.44), (0.57, 0.36), (0.45, 0.36), (0.36, 0.43),
            (0.33, 0.55), (0.37, 0.68), (0.47, 0.74), (0.58, 0.71),
            (0.65, 0.62),
        ],
        [(0.66, 0.18), (0.66, 0.47), (0.66, 0.74)],
    ],
    "e": [
        [
            (0.33, 0.53), (0.51, 0.53), (0.67, 0.53), (0.66, 0.42),
            (0.57, 0.34), (0.45, 0.34), (0.35, 0.41), (0.32, 0.54),
            (0.36, 0.67), (0.47, 0.74), (0.60, 0.73), (0.67, 0.67),
        ],
    ],
    "f": [
        [
            (0.62, 0.23), (0.55, 0.18), (0.47, 0.21), (0.45, 0.31),
            (0.45, 0.54), (0.45, 0.75),
        ],
        [(0.32, 0.41), (0.46, 0.41), (0.61, 0.41)],
    ],
    "g": [
        [
            (0.64, 0.42), (0.56, 0.34), (0.44, 0.34), (0.35, 0.41),
            (0.32, 0.53), (0.36, 0.64), (0.46, 0.69), (0.57, 0.66),
            (0.64, 0.58),
        ],
        [
            (0.65, 0.35), (0.65, 0.62), (0.64, 0.78), (0.57, 0.87),
            (0.45, 0.89), (0.35, 0.84),
        ],
    ],
    "h": [
        [(0.34, 0.18), (0.34, 0.46), (0.34, 0.74)],
        [
            (0.34, 0.48), (0.42, 0.37), (0.54, 0.35), (0.63, 0.41),
            (0.66, 0.54), (0.66, 0.74),
        ],
    ],
    "i": [
        [(0.50, 0.38), (0.50, 0.56), (0.50, 0.74)],
        [(0.50, 0.23), (0.50, 0.25)],
    ],
    "j": [
        [
            (0.54, 0.38), (0.54, 0.64), (0.52, 0.80), (0.44, 0.88),
            (0.35, 0.85),
        ],
        [(0.54, 0.23), (0.54, 0.25)],
    ],
    "k": [
        [(0.35, 0.18), (0.35, 0.46), (0.35, 0.74)],
        [(0.64, 0.36), (0.49, 0.47), (0.36, 0.55)],
        [(0.45, 0.50), (0.56, 0.62), (0.66, 0.74)],
    ],
    "l": [
        [(0.48, 0.18), (0.48, 0.46), (0.48, 0.68), (0.53, 0.74), (0.59, 0.73)],
    ],
    "m": [
        [(0.22, 0.36), (0.22, 0.55), (0.22, 0.74)],
        [
            (0.22, 0.46), (0.29, 0.37), (0.39, 0.36), (0.47, 0.42),
            (0.49, 0.55), (0.49, 0.74),
        ],
        [
            (0.49, 0.46), (0.56, 0.37), (0.66, 0.36), (0.74, 0.42),
            (0.76, 0.55), (0.76, 0.74),
        ],
    ],
    "n": [
        [(0.34, 0.36), (0.34, 0.55), (0.34, 0.74)],
        [
            (0.34, 0.47), (0.42, 0.37), (0.54, 0.35), (0.63, 0.41),
            (0.66, 0.54), (0.66, 0.74),
        ],
    ],
    "ñ": [
        [(0.34, 0.40), (0.34, 0.57), (0.34, 0.74)],
        [
            (0.34, 0.50), (0.42, 0.41), (0.54, 0.39), (0.63, 0.45),
            (0.66, 0.57), (0.66, 0.74),
        ],
        [
            (0.35, 0.27), (0.43, 0.23), (0.51, 0.27), (0.59, 0.29),
            (0.66, 0.24),
        ],
    ],
    "o": [
        [
            (0.50, 0.34), (0.38, 0.37), (0.31, 0.48), (0.32, 0.62),
            (0.41, 0.72), (0.54, 0.74), (0.65, 0.67), (0.69, 0.54),
            (0.65, 0.40), (0.53, 0.34),
        ],
    ],
    "p": [
        [(0.34, 0.36), (0.34, 0.62), (0.34, 0.88)],
        [
            (0.34, 0.45), (0.42, 0.36), (0.54, 0.35), (0.64, 0.42),
            (0.67, 0.55), (0.63, 0.67), (0.52, 0.73), (0.41, 0.71),
            (0.34, 0.64),
        ],
    ],
    "q": [
        [
            (0.65, 0.44), (0.57, 0.36), (0.45, 0.36), (0.35, 0.43),
            (0.32, 0.55), (0.36, 0.67), (0.46, 0.73), (0.58, 0.70),
            (0.65, 0.61),
        ],
        [(0.66, 0.36), (0.66, 0.62), (0.66, 0.88)],
        [(0.55, 0.78), (0.66, 0.78), (0.77, 0.78)],
    ],
    "r": [
        [(0.38, 0.36), (0.38, 0.55), (0.38, 0.74)],
        [(0.38, 0.48), (0.44, 0.39), (0.54, 0.35), (0.64, 0.38)],
    ],
    "s": [
        [
            (0.63, 0.40), (0.55, 0.34), (0.43, 0.34), (0.35, 0.40),
            (0.37, 0.49), (0.50, 0.53), (0.62, 0.58), (0.64, 0.68),
            (0.55, 0.74), (0.42, 0.74), (0.34, 0.68),
        ],
    ],
    "t": [
        [(0.48, 0.21), (0.48, 0.48), (0.48, 0.68), (0.54, 0.74), (0.62, 0.72)],
        [(0.34, 0.38), (0.48, 0.38), (0.63, 0.38)],
    ],
    "u": [
        [
            (0.34, 0.36), (0.34, 0.58), (0.38, 0.70), (0.48, 0.74),
            (0.58, 0.70), (0.65, 0.60),
        ],
        [(0.66, 0.36), (0.66, 0.55), (0.66, 0.74)],
    ],
    "v": [
        [(0.32, 0.36), (0.41, 0.55), (0.50, 0.74), (0.59, 0.55), (0.68, 0.36)],
    ],
    "w": [
        [
            (0.22, 0.36), (0.30, 0.56), (0.37, 0.74), (0.44, 0.56),
            (0.50, 0.42), (0.56, 0.56), (0.63, 0.74), (0.70, 0.56),
            (0.78, 0.36),
        ],
    ],
    "x": [
        [(0.33, 0.36), (0.50, 0.55), (0.67, 0.74)],
        [(0.67, 0.36), (0.50, 0.55), (0.33, 0.74)],
    ],
    "y": [
        [(0.33, 0.36), (0.42, 0.54), (0.50, 0.71)],
        [(0.67, 0.36), (0.54, 0.63), (0.44, 0.82), (0.36, 0.89), (0.30, 0.87)],
    ],
    "z": [
        [
            (0.33, 0.36), (0.50, 0.36), (0.67, 0.36), (0.50, 0.55),
            (0.33, 0.74), (0.50, 0.74), (0.67, 0.74),
        ],
    ],
}

# utilidades de dibujo vectorial en Canvas

def draw_rounded_rect(
    canvas: tk.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
    fill: str,
    outline: str = "",
    width: int = 1,
    tags: str | tuple[str, ...] | None = None,
):
    # rectángulo con esquinas suaves 
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w < 1 or h < 1:
        return None
    r = max(1.0, min(float(radius), w / 2.0, h / 2.0))
    pts = [
        x1 + r, y1,
        x2 - r, y1,
        x2, y1,
        x2, y1 + r,
        x2, y2 - r,
        x2, y2,
        x2 - r, y2,
        x1 + r, y2,
        x1, y2,
        x1, y2 - r,
        x1, y1 + r,
        x1, y1,
    ]
    return canvas.create_polygon(
        pts,
        smooth=True,
        splinesteps=24,
        fill=fill,
        outline=outline if outline else fill,
        width=width,
        tags=tags,
    )


def load_bundled_fonts() -> list[str]:
    # carga las fuentes utilizadas
    if sys.platform != "win32":
        return []
    fonts_dir = ROOT / "assets" / "fonts"
    if not fonts_dir.is_dir():
        return []
    try:
        gdi32 = ctypes.windll.gdi32
        gdi32.AddFontResourceExW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint,
            ctypes.c_void_p,
        ]
        gdi32.AddFontResourceExW.restype = ctypes.c_int
    except (AttributeError, OSError):
        return []

    loaded = []
    for path in sorted(fonts_dir.glob("*.[ot]tf")):
        try:
            if gdi32.AddFontResourceExW(str(path), 0x10, None):
                loaded.append(str(path))
        except (AttributeError, OSError):
            pass
    return loaded


def draw_wave_mark(canvas: tk.Canvas, cx: float, cy: float, scale: float = 1.0):
    # dibuja el isotipo de onda HMM
    pts_main = [
        (cx - 24 * scale, cy + 5 * scale),
        (cx - 16 * scale, cy - 12 * scale),
        (cx - 9 * scale, cy - 14 * scale),
        (cx - 2 * scale, cy - 1 * scale),
        (cx + 6 * scale, cy + 12 * scale),
        (cx + 14 * scale, cy + 11 * scale),
        (cx + 22 * scale, cy - 4 * scale),
    ]
    flat = [v for xy in pts_main for v in xy]
    stroke_w = max(2, int(round(4.2 * scale)))
    canvas.create_line(
        *flat,
        fill=NAVY,
        width=stroke_w,
        smooth=True,
        capstyle="round",
    )
    r = max(2.2, 3.2 * scale)
    x0, y0 = pts_main[0]
    canvas.create_oval(x0 - r, y0 - r, x0 + r, y0 + r, fill=CYAN, outline="")
    pts_accent = pts_main[4:]
    canvas.create_line(
        *(v for xy in pts_accent for v in xy),
        fill=CYAN,
        width=stroke_w,
        smooth=True,
        capstyle="round",
    )

# componentes reutilizables de interfaz

class RoundedCard(tk.Frame):
    # tarjeta blanca con esquinas redondeadas, borde fino y sombra suave

    def __init__(
        self,
        master,
        *,
        padding: int = 14,
        radius: int = 16,
        bg_outer: str = BG,
        panel_bg: str = PANEL,
        border_color: str = PANEL_BORDER,
        shadow_color: str = PANEL_SHADOW,
        **kwargs,
    ):
        super().__init__(master, bg=bg_outer, bd=0, highlightthickness=0, **kwargs)
        self._padding = int(padding)
        self._radius = int(radius)
        self._panel_bg = panel_bg
        self._border_color = border_color
        self._shadow_color = shadow_color

        self._canvas = tk.Canvas(
            self, bg=bg_outer, bd=0, highlightthickness=0
        )
        self._canvas.pack(fill="both", expand=True)

        self.content = tk.Frame(self._canvas, bg=panel_bg, bd=0, highlightthickness=0)
        self._win = self._canvas.create_window(
            self._padding, self._padding, anchor="nw", window=self.content
        )
        self._canvas.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        w = max(int(event.width), 6)
        h = max(int(event.height), 6)
        p = self._padding
        r = self._radius
        self._canvas.delete("card-bg")

        draw_rounded_rect(
            self._canvas,
            2, 3, w - 2, h - 1,
            r,
            fill=self._shadow_color,
            outline=self._shadow_color,
            tags="card-bg",
        )
        draw_rounded_rect(
            self._canvas,
            1, 1, w - 3, h - 3,
            r,
            fill=self._panel_bg,
            outline=self._border_color,
            width=1,
            tags="card-bg",
        )
        self._canvas.tag_lower("card-bg")
        self._canvas.coords(self._win, p, p)
        self._canvas.itemconfigure(
            self._win,
            width=max(1, int(w - 2 * p - 2)),
            height=max(1, int(h - 2 * p - 2)),
        )


class ActionButton(tk.Canvas):
    # botón redondeado 

    def __init__(
        self,
        master,
        text: str,
        command,
        *,
        icon: str | None = None,
        variant: str = "secondary",
        width: int = 128,
        height: int = 36,
        radius: int = 10,
        parent_bg: str = PANEL,
        font=("Segoe UI", 10, "bold"),
    ):
        super().__init__(
            master,
            width=int(width),
            height=int(height),
            bg=parent_bg,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self._text = text
        self._command = command
        self._icon = icon
        self._radius = int(radius)
        self._font = font
        self._hovered = False

        if variant == "primary":
            self._bg = TEAL
            self._hover = TEAL_HOVER
            self._fg = "#ffffff"
            self._border = TEAL
        else:
            self._bg = BTN_BG
            self._hover = BTN_HOVER
            self._fg = TEXT
            self._border = BTN_BORDER

        self._render()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _on_enter(self, _event=None):
        self._hovered = True
        self._render()

    def _on_leave(self, _event=None):
        self._hovered = False
        self._render()

    def _on_click(self, _event=None):
        if callable(self._command):
            self._command()

    def _render(self):
        self.delete("all")
        w = int(self["width"])
        h = int(self["height"])
        fill = self._hover if self._hovered else self._bg
        draw_rounded_rect(
            self, 1, 1, w - 1, h - 1, self._radius,
            fill=fill, outline=self._border, width=1
        )
        cy = h / 2
        if self._icon == "refresh":
            cx = w / 2 - 34
            self.create_arc(
                cx - 6, cy - 6, cx + 6, cy + 6,
                start=40, extent=280, style="arc",
                outline=TEAL, width=2,
            )
            self.create_polygon(
                cx + 4, cy - 7, cx + 8, cy - 3, cx + 3, cy - 2,
                fill=TEAL, outline="",
            )
            self.create_text(
                cx + 14, cy, text=self._text, anchor="w",
                fill=self._fg, font=self._font,
            )
        elif self._icon == "undo":
            cx = w / 2 - 38
            self.create_line(
                cx + 6, cy, cx - 6, cy,
                fill=TEAL, width=2, capstyle="round",
            )
            self.create_line(
                cx - 2, cy - 4, cx - 6, cy, cx - 2, cy + 4,
                fill=TEAL, width=2, capstyle="round", joinstyle="round",
            )
            self.create_text(
                cx + 13, cy, text=self._text, anchor="w",
                fill=self._fg, font=self._font,
            )
        else:
            self.create_text(
                w / 2, cy, text=self._text,
                fill=self._fg, font=self._font,
            )


class ModeSwitch(tk.Canvas):
    # selector segmentado [ 2D | 3D ] 

    def __init__(self, master, on_change, initial: str = "2d", parent_bg: str = PANEL):
        super().__init__(
            master,
            width=104,
            height=30,
            bg=parent_bg,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self._mode = initial
        self._on_change = on_change
        self._render()
        self.bind("<Button-1>", self._handle_click)

    def set_mode(self, mode: str):
        if mode in ("2d", "3d") and mode != self._mode:
            self._mode = mode
            self._render()

    def _handle_click(self, event):
        target = "2d" if event.x < 52 else "3d"
        if target != self._mode:
            self._mode = target
            self._render()
            if callable(self._on_change):
                self._on_change(target)

    def _render(self):
        self.delete("all")
        w, h = 104, 30
        draw_rounded_rect(
            self, 0, 0, w, h, 9,
            fill="#e6f0ed", outline="#d4e3df", width=1
        )
        if self._mode == "2d":
            draw_rounded_rect(self, 3, 3, 51, h - 3, 7, fill=TEAL, outline=TEAL)
            self.create_text(27, h / 2, text="2D", fill="#ffffff", font=("Segoe UI", 9, "bold"))
            self.create_text(77, h / 2, text="3D", fill=MUTED, font=("Segoe UI", 9, "bold"))
        else:
            draw_rounded_rect(self, 53, 3, w - 3, h - 3, 7, fill=TEAL, outline=TEAL)
            self.create_text(27, h / 2, text="2D", fill=MUTED, font=("Segoe UI", 9, "bold"))
            self.create_text(77, h / 2, text="3D", fill="#ffffff", font=("Segoe UI", 9, "bold"))

# cámara sobre esfera para modo 3D

class SphereCamera:
    def __init__(self):
        self.yaw, self.pitch, self.zoom = 0.0, 0.0, 1.0

    def matrix(self) -> np.ndarray:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        return pitch @ yaw

    def project(self, p, cx: float, cy: float, radius: float):
        x, y, z = self.matrix() @ normalize_sphere(p)
        return cx + float(x) * radius, cy - float(y) * radius, float(z)

    def pick(self, x: float, y: float, cx: float, cy: float, radius: float):
        nx, ny = (x - cx) / radius, -(y - cy) / radius
        r2 = nx * nx + ny * ny
        if r2 > 1.0:
            return None
        camera_p = np.array([nx, ny, math.sqrt(max(0.0, 1.0 - r2))])
        return (self.matrix().T @ camera_p).tolist()



# ventana emergente: guía de trazado

class TracingGuideDialog(tk.Toplevel):
    # modal de Guía de Trazado con cuadrícula, flechas direccionales y pasos

    STROKE_COLORS = [
        "#1478c8",  # trazo 1
        "#7650c7",  # trazo 2
        "#ce397f",  # trazo 3
        "#80b9c8",  # trazo 4
    ]

    def __init__(self, parent: tk.Tk, char: str, font_getter):
        super().__init__(parent)
        self.char = char.lower()
        self._font_getter = font_getter
        self._canvas_size = 360
        self.font_ui = ("Selawik", 10, "normal")
        self.font_ui_bold = ("Selawik", 10, "bold")
        self.font_title = ("Selawik", 12, "bold")
        self.font_letter = self._font_getter(self.char, size=17)

        self.title(f'Guía de Trazado: Letra "{self.char}"')
        self.configure(bg=PANEL)
        self.resizable(False, False)
        self.transient(parent)

        header = tk.Frame(self, bg="#eef6f4", height=44)
        header.pack(fill="x")
        header.pack_propagate(False)

        icon_c = tk.Canvas(header, width=28, height=28, bg="#eef6f4", highlightthickness=0)
        icon_c.pack(side="left", padx=(14, 4), pady=8)
        self._draw_pencil_icon(icon_c, 14, 14, TEAL)

        self.header_label = tk.Label(
            header,
            text=f'Guía de Trazado: Letra "{self.char}"',
            bg="#eef6f4",
            fg=TEXT,
            font=self.font_title,
        )
        self.header_label.pack(side="left", pady=8)

        tk.Frame(self, bg="#d9e7e3", height=1).pack(fill="x")

        sub_row = tk.Frame(self, bg=PANEL)
        sub_row.pack(pady=(12, 10))

        tk.Label(
            sub_row,
            text="Guía de Trazado Sugerido para ",
            bg=PANEL,
            fg=TEXT,
            font=self.font_ui_bold,
        ).pack(side="left")

        self.badge_label = tk.Label(
            sub_row,
            text=f" {self.char} ",
            bg=TEAL_LIGHT,
            fg=TEAL,
            font=self.font_letter,
            padx=6,
            pady=0,
        )
        self.badge_label.pack(side="left", padx=(2, 0))

        grid_wrap = tk.Frame(self, bg="#d4e0dc", padx=1, pady=1)
        grid_wrap.pack(padx=22)

        self.grid_canvas = tk.Canvas(
            grid_wrap,
            width=self._canvas_size,
            height=self._canvas_size,
            bg="#ffffff",
            highlightthickness=0,
            bd=0,
        )
        self.grid_canvas.pack()
        self._render_guide(self._canvas_size)

        self.hint_label = tk.Label(
            self,
            text=f"Siga los números y las flechas para un trazado óptimo de '{self.char}'.",
            bg=PANEL,
            fg=TEXT,
            font=self.font_ui,
        )
        self.hint_label.pack(pady=(14, 14))

        btn = ActionButton(
            self,
            text="Cerrar",
            command=self.destroy,
            variant="primary",
            width=108,
            height=38,
            radius=10,
            parent_bg=PANEL,
            font=self.font_ui_bold,
        )
        btn.pack(pady=(0, 18))

        self.bind("<Escape>", lambda _e: self.destroy())
        self._center_on_parent(parent)

    def set_char(self, char: str):
        # actualiza la ventana reutilizada al seleccionar otra letra
        self.char = char.lower()
        self.font_letter = self._font_getter(self.char, size=17)
        self.title(f'Guía de Trazado: Letra "{self.char}"')
        self.header_label.configure(text=f'Guía de Trazado: Letra "{self.char}"')
        self.badge_label.configure(text=f" {self.char} ", font=self.font_letter)
        self.hint_label.configure(
            text=f"Siga los números y las flechas para un trazado óptimo de '{self.char}'."
        )
        self._render_guide(self._canvas_size)

    def _center_on_parent(self, parent: tk.Tk):
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        x = max(20, px + (pw - w) // 2)
        y = max(20, py + (ph - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    @staticmethod
    def _draw_pencil_icon(c: tk.Canvas, cx: float, cy: float, color: str):
        c.create_line(
            cx - 6, cy + 6, cx + 5, cy - 5,
            fill=color, width=2, capstyle="round"
        )
        c.create_polygon(
            cx - 7, cy + 7, cx - 3, cy + 6, cx - 6, cy + 3,
            fill=color, outline=""
        )
        c.create_line(
            cx + 3, cy - 7, cx + 7, cy - 3,
            fill=color, width=2, capstyle="round"
        )

    @staticmethod
    def _draw_step_badge(c: tk.Canvas, cx: float, cy: float, step: int, color: str):
        # dibuja un círculo relleno con el número del trazo centrado
        r = 13.0
        c.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=color, outline="#ffffff", width=2,
        )
        c.create_text(
            cx, cy,
            text=str(step),
            fill="#ffffff",
            font=("Selawik", 11, "bold"),
        )

    def _render_guide(self, size: int):
        c = self.grid_canvas
        c.delete("all")
        step = 20
        for pos in range(step, size, step):
            c.create_line(pos, 0, pos, size, fill="#e4ece9", width=1)
            c.create_line(0, pos, size, pos, fill="#e4ece9", width=1)

        strokes = TRACING_GUIDES.get(self.char, TRACING_GUIDES["a"])
        pad = 24.0
        span = size - 2 * pad

        def to_px(pt: tuple[float, float]) -> tuple[float, float]:
            return (pad + pt[0] * span, pad + pt[1] * span)

        # 1. Trazos base de la letra
        for stroke in strokes:
            px_pts = [to_px(p) for p in stroke]
            if len(px_pts) > 1:
                c.create_line(
                    *(v for xy in px_pts for v in xy),
                    fill="#0d1112",
                    width=10,
                    smooth=True,
                    splinesteps=24,
                    capstyle="round",
                    joinstyle="round",
                )
            elif px_pts:
                x0, y0 = px_pts[0]
                c.create_oval(x0 - 5, y0 - 5, x0 + 5, y0 + 5, fill="#0d1112", outline="")

        badges_to_draw: list[tuple[float, float, int, str]] = []

        # 2. Líneas punteadas, flechas y puntos de inicio con el color de cada trazo
        for idx, stroke in enumerate(strokes, start=1):
            color = self.STROKE_COLORS[(idx - 1) % len(self.STROKE_COLORS)]
            px_pts = [to_px(p) for p in stroke]
            sx, sy = px_pts[0]

            if len(px_pts) >= 2:
                guide_pts = self._offset_guide_curve(px_pts, offset=20.0)
                if len(guide_pts) >= 2:
                    c.create_line(
                        *(v for xy in guide_pts for v in xy),
                        fill=color,
                        width=3,
                        smooth=True,
                        arrow=tk.LAST,
                        arrowshape=(11, 13, 5),
                        capstyle="round",
                        dash=(6, 3),
                    )
                    lbl_idx = min(len(guide_pts) - 1, max(0, len(guide_pts) // 3))
                    lx, ly = guide_pts[lbl_idx]
                    vx, vy = lx - size / 2, ly - size / 2
                    norm = math.hypot(vx, vy) or 1.0
                    lx = max(22, min(size - 22, lx + (vx / norm) * 16))
                    ly = max(22, min(size - 22, ly + (vy / norm) * 16))
                    badges_to_draw.append((lx, ly, idx, color))
            else:
                bx = min(size - 24, sx + 24)
                by = max(22, sy - 6)
                badges_to_draw.append((bx, by, idx, color))

            r = 8.5
            c.create_oval(
                sx - r, sy - r, sx + r, sy + r,
                fill=color, outline="#ffffff", width=1.5,
            )

        # 3. Círculos numerados al frente
        for bx, by, idx, color in badges_to_draw:
            self._draw_step_badge(c, bx, by, idx, color)

    @staticmethod
    def _offset_guide_curve(
        pts: list[tuple[float, float]], offset: float = 20.0
    ) -> list[tuple[float, float]]:
        """Genera una curva paralela recortada (14%..84%) para la flecha direccional."""
        if len(pts) < 2:
            return pts
        if len(pts) >= 5:
            sub = pts[1:-1]
        else:
            p0, p1 = pts[0], pts[-1]
            sub = [
                (p0[0] + (p1[0] - p0[0]) * 0.14, p0[1] + (p1[1] - p0[1]) * 0.14),
                (p0[0] + (p1[0] - p0[0]) * 0.84, p0[1] + (p1[1] - p0[1]) * 0.84),
            ]

        cx, cy = 180.0, 180.0
        result_pos = []
        result_neg = []
        for i, (x, y) in enumerate(sub):
            p_prev = sub[max(0, i - 1)]
            p_next = sub[min(len(sub) - 1, i + 1)]
            dx, dy = p_next[0] - p_prev[0], p_next[1] - p_prev[1]
            length = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / length, dx / length
            result_pos.append((x + nx * offset, y + ny * offset))
            result_neg.append((x - nx * offset, y - ny * offset))

        dist_pos = sum(math.hypot(x - cx, y - cy) for x, y in result_pos)
        dist_neg = sum(math.hypot(x - cx, y - cy) for x, y in result_neg)
        chosen = result_pos if dist_pos >= dist_neg else result_neg
        return [
            (max(16.0, min(344.0, x)), max(16.0, min(344.0, y)))
            for x, y in chosen
        ]

# aplicación principal HMM 2D / 3D

class HMMApp:
    def __init__(self):
        self._private_fonts = load_bundled_fonts()
        self.root = tk.Tk()
        self.root.title("Reconocedor de Caracteres con HMM · 2D / 3D")
        self.root.geometry("1280x760")
        self.root.minsize(1060, 660)
        self.root.configure(bg=BG)

        # carga de modelos HMM y temperaturas de calibración
        self.models: dict[str, dict[str, HMM]] = {"2d": {}, "3d": {}}
        self.warnings: list[str] = []
        self.temperatures = {"2d": 1.0, "3d": 1.0}

        for mode in self.models:
            cal_path = ROOT / "models" / mode / "calibration.json"
            if cal_path.is_file():
                try:
                    cal = json.loads(cal_path.read_text(encoding="utf-8"))
                    temp = float(cal["temperature"])
                    if math.isfinite(temp) and temp > 0:
                        self.temperatures[mode] = temp
                    else:
                        self.warnings.append(
                            f"Temperatura {mode.upper()} inválida; se usará T=1."
                        )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.warnings.append(
                        f"Calibración {mode.upper()} no cargada: {exc}"
                    )

        for mode in self.models:
            for c in LETTERS:
                filename = ROOT / "models" / mode / f"{c}.json"
                if filename.is_file():
                    try:
                        self.models[mode][c] = HMM.load(filename)
                    except Exception as exc:
                        self.warnings.append(f"{mode} {c}: {exc}")

        self.dataset_count = self._count_dataset_samples()

        # estado interactivo del lienzo e inferencia
        self.mode = "2d"
        self._mode_info = self.models["2d"]
        self.strokes: list[list[list[float]]] = []
        self.prediction: list[tuple[str, float, float]] = []
        self.future: list = []
        self.sphere = SphereCamera()
        self.drag_rotate: tuple[int, int] | None = None
        self._inference_id: str | None = None
        self._guide_dialog: TracingGuideDialog | None = None
        self._is_drawing: bool = False
        self.last_observations: list[int] = []
        self.start_time = time.monotonic()

        # referencias a widgets actualizables
        self.canvas: tk.Canvas | None = None
        self.tab_canvas: tk.Canvas | None = None
        self.mode_switch: ModeSwitch | None = None
        self.pred_badge_canvas: tk.Canvas | None = None
        self.conf_value_label: tk.Label | None = None
        self.conf_bar_canvas: tk.Canvas | None = None
        self.prob_rows: list[tuple[tk.Label, tk.Canvas, tk.Label]] = []
        self.prob_values: list[tuple[str, float, bool]] = [
            (ch, 0.0, False) for ch in SPANISH_ALPHABET
        ]
        self.status_label: tk.Label | None = None
        self.hint_3d_frame: tk.Frame | None = None
        self._current_confidence = 0.0
        self._current_best_char = "—"

        self._build_ui()
        self._update_prediction_ui([])
        self.redraw()

    def _count_dataset_samples(self) -> int:
        path = ROOT / "data/raw/glifo-dataset-2026-11-04.json"
        if not path.is_file():
            return 2700
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            total = sum(
                len(v) for v in data.values() if isinstance(v, list)
            )
            return total if total > 0 else 2700
        except (OSError, ValueError, TypeError):
            return 2700

    @staticmethod
    def _letter_font(char: str, size: int = 13, bold: bool = False):
        ch = (char or "").lower()
        family = "Playwrite AR" if ch in ("a", "q") else "Selawik"
        return (family, int(size), "bold" if bold else "normal")

    
    # interfaz principal
    
    def _build_ui(self):
        header = tk.Frame(self.root, bg=BG_HEADER, height=58)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        hdr_left = tk.Frame(header, bg=BG_HEADER)
        hdr_left.pack(side="left", padx=(24, 12), pady=6)

        hdr_logo = tk.Canvas(
            hdr_left, width=54, height=40, bg=BG_HEADER, highlightthickness=0
        )
        hdr_logo.pack(side="left", padx=(0, 8))
        draw_wave_mark(hdr_logo, 27, 20, scale=0.88)

        hdr_titles = tk.Frame(hdr_left, bg=BG_HEADER)
        hdr_titles.pack(side="left")
        tk.Label(
            hdr_titles,
            text="Reconocedor de Caracteres con HMM",
            bg=BG_HEADER,
            fg=TEXT,
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w")
        tk.Label(
            hdr_titles,
            text="Cadenas Ocultas de Markov · Trayectorias 2D / 3D",
            bg=BG_HEADER,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        badge_canvas = tk.Canvas(
            header, width=192, height=34, bg=BG_HEADER, highlightthickness=0
        )
        badge_canvas.pack(side="right", padx=24, pady=12)
        draw_rounded_rect(
            badge_canvas, 1, 1, 191, 33, 10,
            fill="#ffffff", outline=PANEL_BORDER, width=1
        )
        bx, by = 18, 17
        badge_canvas.create_oval(bx - 6, by - 7, bx + 6, by - 3, outline=MUTED, width=1)
        badge_canvas.create_arc(
            bx - 6, by - 3, bx + 6, by + 2, start=180, extent=180,
            style="arc", outline=MUTED, width=1
        )
        badge_canvas.create_arc(
            bx - 6, by + 2, bx + 6, by + 7, start=180, extent=180,
            style="arc", outline=MUTED, width=1
        )
        badge_canvas.create_line(bx - 6, by - 5, bx - 6, by + 5, fill=MUTED, width=1)
        badge_canvas.create_line(bx + 6, by - 5, bx + 6, by + 5, fill=MUTED, width=1)
        badge_canvas.create_text(
            32, 17,
            text=f"Dataset: {self.dataset_count:,} muestras",
            anchor="w",
            fill=TEXT,
            font=("Segoe UI", 9, "bold"),
        )

        tk.Frame(self.root, bg="#cfe4de", height=1).pack(fill="x")

        workspace = tk.Frame(self.root, bg=BG)
        workspace.pack(fill="both", expand=True, padx=16, pady=(12, 14))

        bottom_card = RoundedCard(workspace, padding=12, radius=16, height=96)
        bottom_card.pack(side="bottom", fill="x", pady=(10, 0))
        bottom_card.pack_propagate(False)
        self._build_bottom_guide_bar(bottom_card.content)

        columns = tk.Frame(workspace, bg=BG)
        columns.pack(side="top", fill="both", expand=True)

        left_col = tk.Frame(columns, bg=BG, width=215)
        left_col.pack(side="left", fill="y", padx=(0, 12))
        left_col.pack_propagate(False)
        self._build_left_column(left_col)

        right_col = tk.Frame(columns, bg=BG, width=335)
        right_col.pack(side="right", fill="y", padx=(12, 0))
        right_col.pack_propagate(False)
        self._build_right_column(right_col)

        center_card = RoundedCard(columns, padding=14, radius=16)
        center_card.pack(side="left", fill="both", expand=True)
        self._build_center_column(center_card.content)

    
    # marca HMM + Leyenda de Trayectoria
    
    def _build_left_column(self, parent: tk.Frame):
        brand_box = tk.Frame(parent, bg=BG)
        brand_box.pack(fill="x", pady=(4, 14))

        row_logo = tk.Frame(brand_box, bg=BG)
        row_logo.pack(anchor="center")

        logo_c = tk.Canvas(row_logo, width=54, height=34, bg=BG, highlightthickness=0)
        logo_c.pack(side="left", padx=(0, 4))
        draw_wave_mark(logo_c, 27, 18, scale=0.82)

        tk.Label(
            row_logo,
            text="HMM",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 18, "bold"),
        ).pack(side="left")

        tk.Label(
            brand_box,
            text="Reconocimiento de Letras\ncon Cadenas Ocultas de Markov",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9),
            justify="center",
        ).pack(pady=(2, 0))

        legend_card = RoundedCard(parent, padding=14, radius=16)
        legend_card.pack(fill="x")
        leg = legend_card.content

        tk.Label(
            leg,
            text="Leyenda de Trayectoria",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        tk.Frame(leg, bg="#e5efec", height=1).pack(fill="x", pady=(0, 10))

        legend_items = [
            ("real", "Trayectoria real"),
            ("est", "Trayectoria estimada"),
            ("start", "Punto inicial"),
            ("end", "Punto final"),
        ]
        for kind, label_txt in legend_items:
            row = tk.Frame(leg, bg=PANEL)
            row.pack(fill="x", pady=5)
            ic = tk.Canvas(row, width=32, height=18, bg=PANEL, highlightthickness=0)
            ic.pack(side="left", padx=(0, 8))
            if kind == "real":
                ic.create_line(4, 9, 28, 9, fill=NAVY, width=4, capstyle="round")
            elif kind == "est":
                ic.create_line(
                    4, 9, 28, 9, fill=ESTIMATED, width=3, dash=(4, 4), capstyle="round"
                )
            elif kind == "start":
                ic.create_oval(9, 2, 23, 16, fill=CYAN, outline="")
            elif kind == "end":
                ic.create_oval(9, 2, 23, 16, fill=CORAL, outline="")

            tk.Label(
                row,
                text=label_txt,
                bg=PANEL,
                fg=TEXT,
                font=("Segoe UI", 10),
            ).pack(side="left")

        self.hint_3d_frame = tk.Frame(leg, bg=PANEL)
        tk.Frame(self.hint_3d_frame, bg="#e5efec", height=1).pack(fill="x", pady=(10, 8))
        tk.Label(
            self.hint_3d_frame,
            text="Controles Esfera 3D:\n• Clic izq: dibujar trazo\n• Clic der: rotar esfera\n• Rueda: acercar / alejar",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
            justify="left",
        ).pack(anchor="w", pady=(0, 8))
        ActionButton(
            self.hint_3d_frame,
            text="⟳ Reorientar esfera",
            command=self.reset_camera,
            variant="secondary",
            width=165,
            height=32,
            radius=8,
            parent_bg=PANEL,
        ).pack(anchor="w")

    
    # pestaña dibujo 2D/3D, cuadrícula y botones Limpiar/Deshacer
    
    def _build_center_column(self, parent: tk.Frame):
        top_bar = tk.Frame(parent, bg=PANEL)
        top_bar.pack(fill="x", pady=(0, 8))

        self.tab_canvas = tk.Canvas(
            top_bar, width=148, height=34, bg=PANEL, highlightthickness=0
        )
        self.tab_canvas.pack(side="left")
        self._render_mode_tab()

        right_switch = tk.Frame(top_bar, bg=PANEL)
        right_switch.pack(side="right")

        tk.Label(
            right_switch,
            text="Modo:",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left", padx=(0, 8))

        self.mode_switch = ModeSwitch(
            right_switch, on_change=self.switch_mode, initial=self.mode, parent_bg=PANEL
        )
        self.mode_switch.pack(side="left")

        canvas_frame = tk.Frame(parent, bg="#d7e6e1", padx=1, pady=1)
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg=CANVAS_BG,
            highlightthickness=0,
            bd=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self.mouse_start)
        self.canvas.bind("<B1-Motion>", self.mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_finish)
        self.canvas.bind("<ButtonPress-3>", self.rotate_start)
        self.canvas.bind("<B3-Motion>", self.rotate_move)
        self.canvas.bind("<MouseWheel>", self.mouse_zoom)
        self.canvas.bind("<Button-4>", lambda _e: self.zoom_by(1.08))
        self.canvas.bind("<Button-5>", lambda _e: self.zoom_by(1 / 1.08))

        footer_bar = tk.Frame(parent, bg=PANEL)
        footer_bar.pack(fill="x", pady=(10, 0))

        center_btns = tk.Frame(footer_bar, bg=PANEL)
        center_btns.pack(side="left", expand=True)
        center_btns.pack(fill="x", anchor="w")

        ActionButton(
            center_btns,
            text="Limpiar",
            command=self.clear_canvas,
            icon="refresh",
            variant="secondary",
            width=126,
            height=36,
            radius=10,
            parent_bg=PANEL,
        ).pack(side="left", padx=(0, 12), anchor="w")

        ActionButton(
            center_btns,
            text="Deshacer",
            command=self.undo_stroke,
            icon="undo",
            variant="secondary",
            width=132,
            height=36,
            radius=10,
            parent_bg=PANEL,
        ).pack(side="left", anchor="w")

        self.status_label = tk.Label(
            footer_bar,
            text="0 trazos",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        )
        self.status_label.pack(side="right", padx=(4, 2))

    def _render_mode_tab(self):
        if not self.tab_canvas:
            return
        c = self.tab_canvas
        c.delete("all")
        draw_rounded_rect(c, 0, 1, 146, 33, 9, fill=TEAL, outline=TEAL)
        label = "Dibujo (2D)" if self.mode == "2d" else "Esfera (3D)"
        c.create_text(
            38, 17, text=label, anchor="w",
            fill="#ffffff", font=("Segoe UI", 10, "bold")
        )

    # tarjeta predicción + tarjeta probabilidades (27 letras)
    
    def _build_right_column(self, parent: tk.Frame):
        pred_card = RoundedCard(parent, padding=14, radius=16, height=132)
        pred_card.pack(side="top", fill="x", pady=(0, 10))
        pred_card.pack_propagate(False)
        p_box = pred_card.content

        p_hdr = tk.Frame(p_box, bg=PANEL)
        p_hdr.pack(fill="x", pady=(0, 8))

        eye_c = tk.Canvas(p_hdr, width=22, height=18, bg=PANEL, highlightthickness=0)
        eye_c.pack(side="left", padx=(0, 6))
        eye_c.create_oval(3, 3, 19, 15, outline=TEAL, width=2)
        eye_c.create_oval(8, 6, 14, 12, fill=TEAL, outline="")

        tk.Label(
            p_hdr,
            text="Predicción",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

        p_body = tk.Frame(p_box, bg=PANEL)
        p_body.pack(fill="both", expand=True)

        self.pred_badge_canvas = tk.Canvas(
            p_body, width=68, height=58, bg=PANEL, highlightthickness=0
        )
        self.pred_badge_canvas.pack(side="left", padx=(0, 14))

        conf_col = tk.Frame(p_body, bg=PANEL)
        conf_col.pack(side="left", fill="both", expand=True, pady=(4, 0))

        conf_text_row = tk.Frame(conf_col, bg=PANEL)
        conf_text_row.pack(anchor="w", pady=(0, 8))

        tk.Label(
            conf_text_row,
            text="Confianza: ",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10),
        ).pack(side="left")

        self.conf_value_label = tk.Label(
            conf_text_row,
            text="0.0 %",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 14, "bold"),
        )
        self.conf_value_label.pack(side="left")

        self.conf_bar_canvas = tk.Canvas(
            conf_col, height=13, bg=PANEL, highlightthickness=0
        )
        self.conf_bar_canvas.pack(fill="x")
        self.conf_bar_canvas.bind(
            "<Configure>",
            lambda _e: self._draw_confidence_bar(self._current_confidence),
        )

        prob_card = RoundedCard(parent, padding=14, radius=16)
        prob_card.pack(side="top", fill="both", expand=True)
        pr_box = prob_card.content

        pr_hdr = tk.Frame(pr_box, bg=PANEL)
        pr_hdr.pack(fill="x", pady=(0, 6))

        bars_ic = tk.Canvas(pr_hdr, width=20, height=18, bg=PANEL, highlightthickness=0)
        bars_ic.pack(side="left", padx=(0, 6))
        bars_ic.create_rectangle(3, 9, 6, 15, fill=TEAL, outline="")
        bars_ic.create_rectangle(8, 4, 11, 15, fill=TEAL, outline="")
        bars_ic.create_rectangle(13, 7, 16, 15, fill=TEAL, outline="")

        tk.Label(
            pr_hdr,
            text="Probabilidades (27 letras)",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

        tk.Frame(pr_box, bg="#e5efec", height=1).pack(fill="x", pady=(0, 6))

        scroll_wrap = tk.Frame(pr_box, bg=PANEL)
        scroll_wrap.pack(fill="both", expand=True)

        self._prob_canvas = tk.Canvas(
            scroll_wrap, bg=PANEL, highlightthickness=0, bd=0
        )
        scrollbar = ttk.Scrollbar(
            scroll_wrap, orient="vertical", command=self._prob_canvas.yview
        )
        self._prob_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self._prob_canvas.pack(side="left", fill="both", expand=True)

        self._prob_inner = tk.Frame(self._prob_canvas, bg=PANEL)
        self._prob_win = self._prob_canvas.create_window(
            (0, 0), window=self._prob_inner, anchor="nw"
        )

        self._prob_inner.bind(
            "<Configure>",
            lambda _e: self._prob_canvas.configure(
                scrollregion=self._prob_canvas.bbox("all")
            ),
        )
        self._prob_canvas.bind(
            "<Configure>",
            lambda e: self._prob_canvas.itemconfigure(
                self._prob_win, width=int(e.width)
            ),
        )

        for widget in (self._prob_canvas, self._prob_inner):
            widget.bind("<Enter>", lambda _e: self._bind_prob_scroll())
            widget.bind("<Leave>", lambda _e: self._unbind_prob_scroll())

        self.prob_rows = []

        for idx in range(len(LETTERS)):
            row = tk.Frame(self._prob_inner, bg=PANEL, height=23)
            row.pack(fill="x", pady=1)
            row.pack_propagate(False)

            lbl_char = tk.Label(
                row,
                text="a",
                width=2,
                anchor="w",
                bg=PANEL,
                fg=TEXT,
                font=("Segoe UI", 10, "bold"),
            )
            lbl_char.pack(side="left", padx=(2, 6))

            lbl_pct = tk.Label(
                row,
                text="0.00 %",
                width=7,
                anchor="e",
                bg=PANEL,
                fg=MUTED,
                font=("Segoe UI", 9, "bold"),
            )
            lbl_pct.pack(side="right", padx=(4, 4))

            bar_c = tk.Canvas(
                row,
                height=11,
                bg=PANEL,
                highlightthickness=0,
            )
            bar_c.pack(side="left", fill="x", expand=True, padx=(0, 4))

            bar_c.bind(
                "<Configure>",
                lambda _e, row_idx=idx: self._redraw_single_prob_bar(row_idx),
            )

            self.prob_rows.append((lbl_char, bar_c, lbl_pct))

    def _bind_prob_scroll(self):
        self.root.bind_all("<MouseWheel>", self._on_prob_wheel)
        self.root.bind_all("<Button-4>", self._on_prob_wheel)
        self.root.bind_all("<Button-5>", self._on_prob_wheel)

    def _unbind_prob_scroll(self):
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_prob_wheel(self, event):
        if not hasattr(self, "_prob_canvas") or not self._prob_canvas.winfo_exists():
            return
        step = (
            -2
            if getattr(event, "num", None) == 4
            else 2
            if getattr(event, "num", None) == 5
            else (-2 if event.delta > 0 else 2)
        )
        self._prob_canvas.yview_scroll(int(step), "units")

    # guía de Trazado de Caracteres (las 27 letras)
    
    def _build_bottom_guide_bar(self, parent: tk.Frame):
        left_info = tk.Frame(parent, bg=PANEL, width=370)
        left_info.pack(side="left", fill="y", padx=(4, 16))
        left_info.pack_propagate(False)

        title_row = tk.Frame(left_info, bg=PANEL)
        title_row.pack(anchor="w", pady=(2, 4))

        gear_c = tk.Canvas(title_row, width=20, height=20, bg=PANEL, highlightthickness=0)
        gear_c.pack(side="left", padx=(0, 6))
        gear_c.create_oval(4, 4, 16, 16, outline=TEAL, width=2)
        gear_c.create_oval(8, 8, 12, 12, fill=TEAL, outline="")

        tk.Label(
            title_row,
            text="Guía de Trazado de Caracteres",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

        tk.Label(
            left_info,
            text="Pase el cursor por una letra para resaltar y haga clic para ver su guía.",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
            wraplength=350,
            justify="left",
        ).pack(anchor="w")

        tiles_grid = tk.Frame(parent, bg=PANEL)
        tiles_grid.pack(side="right", fill="both", expand=True)

        for idx, letter in enumerate(SPANISH_ALPHABET):
            r = 0 if idx < 14 else 1
            c = idx if idx < 14 else idx - 14
            tiles_grid.columnconfigure(c, weight=1)
            tiles_grid.rowconfigure(r, weight=1)

            tile = tk.Canvas(
                tiles_grid,
                width=40,
                height=31,
                bg=PANEL,
                highlightthickness=0,
                cursor="hand2",
            )
            tile.grid(row=r, column=c, padx=3, pady=2, sticky="nsew")
            self._configure_letter_tile(tile, letter)

    def _configure_letter_tile(self, tile: tk.Canvas, letter: str):
        state = {"hover": False}

        def render(_event=None):
            tile.delete("all")
            w = max(int(tile.winfo_width()), 34)
            h = max(int(tile.winfo_height()), 28)
            fill = TILE_HOVER if state["hover"] else TILE_BG
            border = TEAL if state["hover"] else "#b3c7d4"
            draw_rounded_rect(
                tile, 1, 1, w - 1, h - 1, 7,
                fill=fill, outline=border, width=1
            )
            tile.create_text(
                w / 2,
                h / 2,
                text=letter.lower(),
                fill="#ffffff" if state["hover"] else TEXT,
                font=self._letter_font(letter, size=12, bold=False),
            )

        def on_enter(_e):
            state["hover"] = True
            render()

        def on_leave(_e):
            state["hover"] = False
            render()

        tile.bind("<Configure>", render)
        tile.bind("<Enter>", on_enter)
        tile.bind("<Leave>", on_leave)
        tile.bind("<Button-1>", lambda _e, ch=letter: self.open_tracing_guide(ch))

    def open_tracing_guide(self, char: str):
        if self._guide_dialog is not None and self._guide_dialog.winfo_exists():
            self._guide_dialog.set_char(char)
            self._guide_dialog.deiconify()
            self._guide_dialog.lift()
            return
        self._guide_dialog = TracingGuideDialog(self.root, char, self._letter_font)

    # cambio de modo 2D / 3D y acciones de dibujo
    
    def switch_mode(self, new_mode: str):
        if new_mode not in ("2d", "3d") or new_mode == self.mode:
            return
        self.mode = new_mode
        self._mode_info = self.models[new_mode]
        self._render_mode_tab()

        if self.hint_3d_frame is not None:
            if new_mode == "3d":
                self.hint_3d_frame.pack(fill="x", pady=(4, 0))
            else:
                self.hint_3d_frame.pack_forget()

        self.clear_canvas()
        self.sphere = SphereCamera()
        self.redraw()

    def clear_canvas(self):
        if self._inference_id is not None:
            self.root.after_cancel(self._inference_id)
            self._inference_id = None
        self._is_drawing = False
        self.strokes = []
        self.future = []
        self.prediction = []
        self.last_observations = []
        self._update_prediction_ui([])
        self.redraw()

    def undo_stroke(self):
        """Elimina el último trazo dibujado y recalcula la predicción HMM."""
        if not self.strokes:
            return
        self._is_drawing = False
        self.strokes.pop()
        if not self.strokes:
            self.clear_canvas()
            return
        self.future = []
        self.redraw()
        self.run_inference()

    def reset_camera(self):
        self.sphere = SphereCamera()
        self.redraw()

    # captura de eventos del ratón en el lienzo 2D / 3D

    def _sphere_dimensions(self):
        w = max(int(self.canvas.winfo_width()), 2)
        h = max(int(self.canvas.winfo_height()), 2)
        return w / 2, h / 2, min(w, h) * 0.40 * self.sphere.zoom

    def _point_from_event(self, event):
        if self.mode == "2d":
            w = max(int(self.canvas.winfo_width()), 2)
            h = max(int(self.canvas.winfo_height()), 2)
            x = max(0, min(w, event.x)) / w * 1000.0
            y = max(0, min(h, event.y)) / h * 1000.0
            return [x, y, (time.monotonic() - self.start_time) * 1000.0]

        cx, cy, r = self._sphere_dimensions()
        pos = self.sphere.pick(event.x, event.y, cx, cy, r)
        if pos is None:
            return None
        return pos + [(time.monotonic() - self.start_time) * 1000.0]

    def mouse_start(self, event):
        if not self.strokes:
            self.start_time = time.monotonic()
        p = self._point_from_event(event)
        if p is None:
            return
        self._is_drawing = True
        self.strokes.append([p])
        self.redraw()
        self.schedule_inference()

    def mouse_move(self, event):
        if not self.strokes:
            return
        p = self._point_from_event(event)
        if p is None:
            return
        last = self.strokes[-1][-1]
        dist = (
            math.dist(p[:2], last[:2])
            if self.mode == "2d"
            else math.dist(p[:3], last[:3]) * 500.0
        )
        if dist < 3.0:
            return
        self._is_drawing = True
        self.strokes[-1].append(p)
        self.redraw()
        self.schedule_inference()

    def mouse_finish(self, _event):
        self._is_drawing = False
        if self._inference_id is not None:
            self.root.after_cancel(self._inference_id)
            self._inference_id = None
        self.run_inference()

    def rotate_start(self, event):
        self.drag_rotate = (event.x, event.y)

    def rotate_move(self, event):
        if self.mode != "3d" or self.drag_rotate is None:
            return
        dx = event.x - self.drag_rotate[0]
        dy = event.y - self.drag_rotate[1]
        self.sphere.yaw += dx * 0.009
        self.sphere.pitch = max(-1.45, min(1.45, self.sphere.pitch + dy * 0.009))
        self.drag_rotate = (event.x, event.y)
        self.redraw()

    def zoom_by(self, factor: float):
        self.sphere.zoom = min(1.22, max(0.55, self.sphere.zoom * factor))
        self.redraw()

    def mouse_zoom(self, event):
        if self.mode == "3d":
            self.zoom_by(1.08 if event.delta > 0 else 1 / 1.08)

    # clasificación bayesiana sensible a trazos terminados y continuaciones
    
    @staticmethod
    def _stroke_priors_for(
        char: str, model: HMM
    ) -> tuple[dict[int, float], dict[int, float], float]:
        # combina priors empíricos de trazos con la estructura canónica de la letra
        ch = char.lower()
        canonical = len(TRACING_GUIDES.get(ch, [[(0.0, 0.0)]]))
        meta = getattr(model, "metadata", None) or {}
        raw_probs = meta.get("stroke_count_probs") if isinstance(meta, dict) else None

        emp: dict[int, float] = {}
        if isinstance(raw_probs, dict) and raw_probs:
            for k_str, v in raw_probs.items():
                try:
                    emp[int(k_str)] = float(v)
                except (TypeError, ValueError):
                    pass

        s_probs: dict[int, float] = {}
        for k in (1, 2, 3, 4):
            if ch == "q":
                cp = 0.56 if k == 2 else (0.36 if k == 3 else 0.04)
            else:
                cp = 0.92 if k == canonical else (0.04 if abs(k - canonical) == 1 else 0.01)
            ep = emp.get(k, cp if emp else cp)
            s_probs[k] = (0.60 * ep + 0.40 * cp) if emp else cp
        tot = sum(s_probs.values()) or 1.0
        s_probs = {k: v / tot for k, v in s_probs.items()}

        len_map: dict[int, float] = {}
        raw_lens = meta.get("length_by_strokes") if isinstance(meta, dict) else None
        if isinstance(raw_lens, dict):
            for k_str, v in raw_lens.items():
                try:
                    len_map[int(k_str)] = float(v)
                except (TypeError, ValueError):
                    pass

        base_len = max(8.0, float(getattr(model, "median_length", 22) or 22))
        for k in (1, 2, 3, 4):
            if k not in len_map:
                len_map[k] = base_len * (0.88 + 0.12 * (k / max(canonical, 1)))

        return s_probs, len_map, base_len

    def _classify_with_stroke_context(
        self, obs: list[int]
    ) -> list[tuple[str, float, float]]:
        # evalúa cada letra considerando si los trazos actuales la completan o son un prefijo
        models = self._mode_info
        if not models:
            return []

        obs_len = max(1, len(obs))
        dragged_strokes = sum(1 for st in self.strokes if len(st) >= 2)
        total_strokes = len(self.strokes)
        is_drawing = self._is_drawing

        adjusted_scores: dict[str, float] = {}
        raw_scores: dict[str, float] = {}

        for ch, model in models.items():
            raw_ll = float(model.forward(obs)[2])
            raw_scores[ch] = raw_ll

            # para 'i' y 'j' un punto (1 coordenada) cuenta como trazo válido (el punto superior)
            s_obs = max(
                1,
                total_strokes if ch in ("i", "j") else (dragged_strokes or total_strokes),
            )
            s_probs, len_map, med_len = self._stroke_priors_for(ch, model)

            try:
                gamma_last = model.posterior(obs)[-1]
                prog = float(
                    (gamma_last @ np.arange(model.states, dtype=float))
                    / max(model.states - 1, 1)
                )
            except Exception:
                prog = min(1.0, obs_len / max(med_len, 1.0))

            # hipótesis 1: la letra YA ESTÁ COMPLETA con s_obs trazos
            p_complete_prior = s_probs.get(s_obs, 0.01)
            expected_complete_len = max(6.0, float(len_map.get(s_obs, med_len)))
            len_ratio = obs_len / expected_complete_len
            if len_ratio < 0.76:
                len_fit = math.exp(-3.5 * ((0.76 - len_ratio) ** 2) / 0.14)
            elif len_ratio > 1.48:
                len_fit = math.exp(-1.5 * ((len_ratio - 1.48) ** 2) / 0.25)
            else:
                len_fit = 1.0

            state_end_fit = math.exp(-2.5 * (max(0.0, 0.80 - prog) ** 2) / 0.12)
            complete_weight = (
                (0.55 if is_drawing else 1.0)
                * p_complete_prior
                * len_fit
                * state_end_fit
            )

            # hipótesis 2: letra INCOMPLETA que coincide como prefijo y le faltan trazos (ej. 'q' tras 'o')
            cont_weight = 0.0
            for k, pk in s_probs.items():
                if k > s_obs:
                    missing = k - s_obs
                    step_decay = (0.36 if is_drawing else 0.22) ** missing
                    expected_k_len = max(8.0, float(len_map.get(k, med_len)))
                    expected_prefix = expected_k_len * (s_obs / k)
                    prefix_ratio = obs_len / max(expected_prefix, 6.0)
                    prefix_fit = math.exp(
                        -0.8
                        * (max(0.0, abs(math.log(max(prefix_ratio, 0.25))) - 0.35) ** 2)
                    )
                    cont_weight += pk * step_decay * prefix_fit

            excess_weight = 0.0
            for k, pk in s_probs.items():
                if k < s_obs:
                    excess = s_obs - k
                    excess_weight += pk * (0.018 ** excess)

            stroke_likelihood = max(1e-6, complete_weight + cont_weight + excess_weight)

            length_bias = 0.11 * obs_len * math.log(max(med_len, 8.0) / 20.0)
            adjusted_scores[ch] = raw_ll - length_bias + math.log(stroke_likelihood)

        bg_score = obs_len * -2.65

        base_temp = max(float(self.temperatures.get(self.mode, 1.0)), 1e-3)
        eff_temp = base_temp * max(1.0, obs_len / 28.0)

        max_adj = max(max(adjusted_scores.values()), bg_score)
        exps = {
            ch: math.exp((val - max_adj) / eff_temp)
            for ch, val in adjusted_scores.items()
        }
        bg_exp = math.exp((bg_score - max_adj) / eff_temp)
        total_exp = (sum(exps.values()) + bg_exp) or 1.0

        ranking = [
            (ch, exps[ch] / total_exp, raw_scores[ch])
            for ch in adjusted_scores
        ]
        ranking.sort(key=lambda item: item[1], reverse=True)
        return ranking

    # inferencia HMM en tiempo real
    
    def schedule_inference(self):
        if self._inference_id is None:
            self._inference_id = self.root.after(80, self.run_inference)

    def run_inference(self):
        self._inference_id = None
        if not self.strokes or len(self._mode_info) != 27:
            return
        try:
            if self.mode == "2d":
                obs = extract_2d(self.strokes, sectors=8, step_units=10.0)
            else:
                obs = extract_3d(self.strokes, sectors=8, step_units=0.018)

            self.last_observations = obs
            ranking = self._classify_with_stroke_context(obs)
            self.prediction = ranking
            self._update_prediction_ui(ranking)

            best_char = ranking[0][0] if ranking else None
            model = self._mode_info.get(best_char)
            if model:
                current = self.strokes[-1][-1]
                if self.mode == "2d":
                    projected, aligned = self._project_2d_completion(best_char, model, obs)
                    if aligned:
                        self.future = projected
                    else:
                        fallback = project_2d(
                            model, obs, current[:2], sectors=8, horizon=16
                        )
                        self.future = [fallback] if len(fallback) > 1 else []
                else:
                    self.future = project_3d(
                        model, obs, np.array(current[:3]), sectors=8
                    )
            else:
                self.future = []
        except Exception as exc:
            self.future = []
            if self.status_label:
                self.status_label.configure(text=f"Error: {str(exc)[:40]}")
        self.redraw()

    # actualización visual de predicción, confianza y probabilidades (27 letras)
    
    def _draw_prediction_badge(self, char: str):
        if not self.pred_badge_canvas:
            return
        c = self.pred_badge_canvas
        c.delete("all")
        draw_rounded_rect(c, 1, 1, 67, 57, 12, fill=TEAL, outline=TEAL)
        c.create_text(
            34,
            28,
            text=char.lower() if char != "—" else char,
            fill="#ffffff",
            font=self._letter_font(char, 26, bold=False),
        )

    def _draw_confidence_bar(self, prob: float):
        if not self.conf_bar_canvas:
            return
        c = self.conf_bar_canvas
        c.delete("all")
        w = max(int(c.winfo_width()), 120)
        h = 13
        draw_rounded_rect(c, 0, 1, w, h - 1, 6, fill=BAR_TRACK, outline=BAR_TRACK)
        clamped = max(0.0, min(1.0, prob))
        if clamped > 0.005:
            fill_w = max(12.0, w * clamped)
            draw_rounded_rect(c, 0, 1, fill_w, h - 1, 6, fill=TEAL, outline=TEAL)

    def _redraw_single_prob_bar(self, idx: int):
        if idx >= len(self.prob_rows) or idx >= len(self.prob_values):
            return
        _, bar_c, _ = self.prob_rows[idx]
        _, prob, is_top = self.prob_values[idx]
        bar_c.delete("all")
        raw_w = int(bar_c.winfo_width())
        bw = raw_w if raw_w > 1 else 185
        bh = 11
        draw_rounded_rect(
            bar_c, 0, 1, bw, bh - 1, 5, fill=BAR_TRACK, outline=BAR_TRACK
        )
        clamped = max(0.0, min(1.0, prob))
        if clamped > 0.001:
            fw = max(8.0, bw * clamped)
            color = TEAL if is_top else BAR_SECONDARY
            draw_rounded_rect(
                bar_c, 0, 1, fw, bh - 1, 5, fill=color, outline=color
            )

    def _update_prediction_ui(self, ranking: list[tuple[str, float, float]]):
        if ranking:
            best_char, best_prob, _ = ranking[0]
            self._current_best_char = best_char
            self._current_confidence = best_prob
            self._draw_prediction_badge(best_char)
            if self.conf_value_label:
                self.conf_value_label.configure(text=f"{best_prob * 100:.1f} %")
            self._draw_confidence_bar(best_prob)
        else:
            self._current_best_char = "—"
            self._current_confidence = 0.0
            self._draw_prediction_badge("—")
            if self.conf_value_label:
                self.conf_value_label.configure(text="0.0 %")
            self._draw_confidence_bar(0.0)

        ordered = ranking if ranking else [(ch, 0.0, 0.0) for ch in SPANISH_ALPHABET]
        for idx, ((lbl_char, _bar_c, lbl_pct), (char, prob, _)) in enumerate(
            zip(self.prob_rows, ordered)
        ):
            is_top = bool(ranking) and idx == 0
            self.prob_values[idx] = (char, prob, is_top)
            lbl_char.configure(
                text=char.lower(),
                fg=TEAL if is_top else TEXT,
                font=self._letter_font(char, 11, bold=False),
            )
            lbl_pct.configure(
                text=f"{prob * 100:.1f} %",
                fg=TEXT if is_top else MUTED,
            )
            self._redraw_single_prob_bar(idx)

    # renderizado del lienzo central 2D y 3D (con punto inicial y punto final)

    @staticmethod
    def _polyline_length(pts: np.ndarray) -> float:
        if len(pts) < 2:
            return 0.0
        diffs = np.diff(pts, axis=0)
        return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())

    @staticmethod
    def _resample_polyline(pts: np.ndarray, count: int) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=float)
        if len(pts) == 1 or count <= 1:
            return np.repeat(pts[:1], max(1, count), axis=0)
        diffs = np.diff(pts, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
        total = float(cum[-1])
        if total < 1e-6:
            return np.repeat(pts[:1], count, axis=0)
        targets = np.linspace(0.0, total, count)
        return np.column_stack([
            np.interp(targets, cum, pts[:, 0]),
            np.interp(targets, cum, pts[:, 1]),
        ])

    @classmethod
    def _fit_similarity_2d(
        cls,
        src: np.ndarray,
        dst: np.ndarray,
        max_angle_deg: float = 10.0,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        
        mu_src = src.mean(axis=0)
        mu_dst = dst.mean(axis=0)
        A = src - mu_src
        B = dst - mu_dst

        len_src = cls._polyline_length(src)
        len_dst = cls._polyline_length(dst)
        if len_src < 1e-6 or len_dst < 1e-6:
            return 1.0, np.eye(2), mu_src, mu_dst, float("inf")

        scale = float(np.clip(len_dst / len_src, 0.45, 1.65))

        H = A.T @ B
        angle = math.atan2(float(H[0, 1] - H[1, 0]), float(H[0, 0] + H[1, 1]))
        max_rad = math.radians(max_angle_deg)
        angle = max(-max_rad, min(max_rad, angle))
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, s], [-s, c]], dtype=float)

        aligned = scale * (A @ R) + mu_dst
        norm_factor = max(len_dst, 50.0)
        rms_err = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1)))) / norm_factor
        end_err = float(np.linalg.norm(aligned[-1] - dst[-1])) / norm_factor
        rot_penalty = 0.12 * abs(angle)
        return scale, R, mu_src, mu_dst, rms_err + 0.5 * end_err + rot_penalty

    @classmethod
    def _build_continuous_guide(
        cls, guide_strokes: list[np.ndarray]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        # combina trazos conectados de una letra multitrazos y separa diacríticos
        body = [guide_strokes[0]]
        floating: list[np.ndarray] = []
        for st in guide_strokes[1:]:
            gap = float(np.linalg.norm(st[0] - body[-1][-1]))
            st_len = cls._polyline_length(st)
            if gap > 220.0 and st_len < 360.0 and float(np.mean(st[:, 1])) < 360.0:
                floating.append(st)
            else:
                body.append(st)
        return np.vstack(body), floating

    @staticmethod
    def _clamp_to_canvas(
        polylines: list[np.ndarray],
        anchor: np.ndarray,
        margin: float = 28.0,
    ) -> list[np.ndarray]:
        # mantiene toda la proyección dentro de los límites visibles del lienzo [0..1000]
        if not polylines:
            return polylines
        all_pts = np.vstack(polylines)
        min_xy = all_pts.min(axis=0)
        max_xy = all_pts.max(axis=0)
        sx, sy = 1.0, 1.0
        if min_xy[0] < margin and anchor[0] - min_xy[0] > 1e-3:
            sx = min(sx, (anchor[0] - margin) / (anchor[0] - min_xy[0]))
        if max_xy[0] > 1000.0 - margin and max_xy[0] - anchor[0] > 1e-3:
            sx = min(sx, (1000.0 - margin - anchor[0]) / (max_xy[0] - anchor[0]))
        if min_xy[1] < margin and anchor[1] - min_xy[1] > 1e-3:
            sy = min(sy, (anchor[1] - margin) / (anchor[1] - min_xy[1]))
        if max_xy[1] > 1000.0 - margin and max_xy[1] - anchor[1] > 1e-3:
            sy = min(sy, (1000.0 - margin - anchor[1]) / (max_xy[1] - anchor[1]))
        s = max(0.35, min(sx, sy))
        if s >= 0.999:
            return polylines
        return [anchor + s * (poly - anchor) for poly in polylines]

    def _project_2d_completion(
        self, best_char: str, model: HMM, obs: list[int]
    ) -> tuple[list[list[tuple[float, float]]], bool]:
        # proyecta desde el último punto escrito cómo quedaría completo el trazo
        char = (best_char or "").lower()
        if char not in TRACING_GUIDES or not self.strokes:
            return [], False

        user_strokes = [
            np.asarray([[p[0], p[1]] for p in st], dtype=float)
            for st in self.strokes
            if len(st) > 0
        ]
        total_pts = sum(len(st) for st in user_strokes)
        if not user_strokes or total_pts < 4:
            return [], False

        guide_strokes = [
            np.asarray(st, dtype=float) * 1000.0 for st in TRACING_GUIDES[char]
        ]
        if not guide_strokes:
            return [], False

        last_point = user_strokes[-1][-1]

        substantive = [st for st in user_strokes if self._polyline_length(st) >= 28.0]
        if not substantive:
            substantive = [np.vstack(user_strokes)]

        n_guide = len(guide_strokes)
        if len(substantive) > n_guide:
            if n_guide == 1:
                substantive = [np.vstack(substantive)]
            else:
                substantive = substantive[: n_guide - 1] + [
                    np.vstack(substantive[n_guide - 1 :])
                ]
        n_user = len(substantive)

        try:
            gamma_last = model.posterior(obs)[-1]
            p_hmm = float(
                (gamma_last @ np.arange(model.states, dtype=float))
                / max(model.states - 1, 1)
            )
            p_hmm = float(np.clip(p_hmm, 0.12, 0.96))
        except Exception:
            p_hmm = 0.5

        candidates: list[tuple[float, list[np.ndarray]]] = []

        # hipótesis 1: correspondencia trazo a trazo
        if n_user <= n_guide:
            active_idx = n_user - 1
            active_guide_dense = self._resample_polyline(guide_strokes[active_idx], 100)
            completed_src = [
                self._resample_polyline(guide_strokes[i], 30) for i in range(active_idx)
            ]
            completed_dst = [
                self._resample_polyline(substantive[i], 30) for i in range(active_idx)
            ]
            active_dst = self._resample_polyline(substantive[active_idx], 36)

            best_h1 = None
            for frac in np.linspace(0.12, 1.00, 45):
                cut = int(
                    np.clip(
                        round(frac * (len(active_guide_dense) - 1)),
                        1,
                        len(active_guide_dense) - 1,
                    )
                )
                prefix_src = self._resample_polyline(active_guide_dense[: cut + 1], 36)

                src_all = (
                    np.vstack(completed_src + [prefix_src])
                    if completed_src
                    else prefix_src
                )
                dst_all = (
                    np.vstack(completed_dst + [active_dst])
                    if completed_dst
                    else active_dst
                )

                scale, R, mu_s, mu_d, err = self._fit_similarity_2d(src_all, dst_all)
                all_guide_pts = np.vstack(guide_strokes)
                full_tr = scale * ((all_guide_pts - mu_s) @ R) + mu_d
                oob = float(
                    np.maximum(0.0, 20.0 - full_tr).sum()
                    + np.maximum(0.0, full_tr - 980.0).sum()
                ) / 800.0

                overall_prog = (active_idx + frac) / n_guide
                reg = (
                    0.18 * abs(overall_prog - p_hmm)
                    + 0.14 * abs(math.log(max(scale, 1e-3)))
                    + 0.30 * oob
                )
                score = err + reg
                if best_h1 is None or score < best_h1[0]:
                    best_h1 = (score, frac, cut, scale, R, mu_s, mu_d)

            if best_h1 is not None:
                score, frac, cut, scale, R, mu_s, mu_d = best_h1
                projected_list: list[np.ndarray] = []

                def to_canvas1(pts: np.ndarray) -> np.ndarray:
                    return scale * ((pts - mu_s) @ R) + mu_d

                anchor_shift = last_point - to_canvas1(active_guide_dense[cut])
                template_end = to_canvas1(active_guide_dense[-1]) + anchor_shift
                already_at_end = float(np.linalg.norm(last_point - template_end)) < 40.0

                suffix = active_guide_dense[cut:]
                if len(suffix) >= 2 and frac < 0.92 and not already_at_end:
                    rem = to_canvas1(suffix) + anchor_shift
                    if self._polyline_length(rem) >= 22.0:
                        projected_list.append(
                            self._resample_polyline(rem, max(16, len(rem) // 2))
                        )

                body_shift = (
                    0.35 * anchor_shift
                    if (frac >= 0.92 or already_at_end)
                    else anchor_shift
                )
                for rem_idx in range(n_user, n_guide):
                    rem_st = guide_strokes[rem_idx]
                    rem_dense = self._resample_polyline(
                        rem_st, max(12, len(rem_st) * 4)
                    )
                    rem_tr = to_canvas1(rem_dense) + body_shift
                    if not projected_list and rem_idx == n_user:
                        gap = float(np.linalg.norm(rem_tr[0] - last_point))
                        guide_gap = float(
                            np.linalg.norm(
                                guide_strokes[rem_idx][0]
                                - guide_strokes[rem_idx - 1][-1]
                            )
                        )
                        if guide_gap < 280.0 and gap < 380.0:
                            retrace = np.linspace(
                                last_point, rem_tr[0], max(4, int(gap / 25.0))
                            )
                            rem_tr = np.vstack([retrace[:-1], rem_tr])
                    projected_list.append(rem_tr)

                projected_list = self._clamp_to_canvas(projected_list, last_point)
                weight_h1 = 0.88 if (n_user == 1 and n_guide > 1) else 1.0
                candidates.append((score * weight_h1, projected_list))

        # hipótesis 2: Letra multitrazos dibujada en un solo trazo continuo
        if n_user == 1 and n_guide > 1:
            cont_guide, floating_diacritics = self._build_continuous_guide(guide_strokes)
            cont_dense = self._resample_polyline(cont_guide, 120)
            user_res = self._resample_polyline(substantive[0], 40)

            best_h2 = None
            for frac in np.linspace(0.12, 1.00, 45):
                cut = int(
                    np.clip(
                        round(frac * (len(cont_dense) - 1)), 1, len(cont_dense) - 1
                    )
                )
                prefix_src = self._resample_polyline(cont_dense[: cut + 1], 40)
                scale, R, mu_s, mu_d, err = self._fit_similarity_2d(prefix_src, user_res)

                all_guide_pts = np.vstack(guide_strokes)
                full_tr = scale * ((all_guide_pts - mu_s) @ R) + mu_d
                oob = float(
                    np.maximum(0.0, 20.0 - full_tr).sum()
                    + np.maximum(0.0, full_tr - 980.0).sum()
                ) / 800.0

                reg = (
                    0.18 * abs(frac - p_hmm)
                    + 0.14 * abs(math.log(max(scale, 1e-3)))
                    + 0.30 * oob
                )
                score = err + reg
                if best_h2 is None or score < best_h2[0]:
                    best_h2 = (score, frac, cut, scale, R, mu_s, mu_d)

            if best_h2 is not None:
                score, frac, cut, scale, R, mu_s, mu_d = best_h2
                projected_list = []

                def to_canvas2(pts: np.ndarray) -> np.ndarray:
                    return scale * ((pts - mu_s) @ R) + mu_d

                anchor_shift = last_point - to_canvas2(cont_dense[cut])
                template_end = to_canvas2(cont_dense[-1]) + anchor_shift
                already_at_end = float(np.linalg.norm(last_point - template_end)) < 40.0

                suffix = cont_dense[cut:]
                if len(suffix) >= 2 and frac < 0.92 and not already_at_end:
                    rem = to_canvas2(suffix) + anchor_shift
                    if self._polyline_length(rem) >= 22.0:
                        projected_list.append(
                            self._resample_polyline(rem, max(16, len(rem) // 2))
                        )

                for dia in floating_diacritics:
                    dia_dense = self._resample_polyline(dia, max(10, len(dia) * 4))
                    dia_tr = to_canvas2(dia_dense) + 0.35 * anchor_shift
                    projected_list.append(dia_tr)

                projected_list = self._clamp_to_canvas(projected_list, last_point)
                candidates.append((score, projected_list))

        if not candidates:
            return [], False

        _, best_polylines = min(candidates, key=lambda item: item[0])
        result = [
            [(float(pt[0]), float(pt[1])) for pt in poly]
            for poly in best_polylines
            if len(poly) >= 1
        ]
        return result, True

    def _draw_2d_scene(self):
        w = max(int(self.canvas.winfo_width()), 2)
        h = max(int(self.canvas.winfo_height()), 2)

        spacing = 24
        for x in range(spacing, w, spacing):
            major = (x % (spacing * 5) == 0)
            self.canvas.create_line(
                x, 0, x, h,
                fill=GRID_MAJOR if major else GRID_MINOR,
                width=1,
            )
        for y in range(spacing, h, spacing):
            major = (y % (spacing * 5) == 0)
            self.canvas.create_line(
                0, y, w, y,
                fill=GRID_MAJOR if major else GRID_MINOR,
                width=1,
            )

        if not self.strokes:
            self.canvas.create_text(
                w / 2,
                h / 2 - 10,
                text="Dibuja una letra aquí",
                fill=MUTED,
                font=("Segoe UI", 15, "bold"),
            )
            self.canvas.create_text(
                w / 2,
                h / 2 + 16,
                text="Usa el ratón para trazar y observa la clasificación HMM en tiempo real",
                fill="#7b8e8a",
                font=("Segoe UI", 10),
            )
            return

        # proyección estimada completa desde el último punto (línea discontinua)
        for poly in self.future:
            if not poly:
                continue
            pts = [(x / 1000.0 * w, y / 1000.0 * h) for x, y in poly]
            span = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
            if len(pts) == 1 or span < 8.0:
                cx_d, cy_d = pts[0]
                self.canvas.create_oval(
                    cx_d - 4.5, cy_d - 4.5, cx_d + 4.5, cy_d + 4.5,
                    fill=ESTIMATED, outline="",
                )
            else:
                self.canvas.create_line(
                    *(v for xy in pts for v in xy),
                    fill=ESTIMATED,
                    width=3,
                    dash=(6, 5),
                    smooth=True,
                    splinesteps=20,
                    capstyle="round",
                    joinstyle="round",
                )

        # trayectoria real (trazo grueso oscuro con acabado suave)
        for stroke in self.strokes:
            pts = [(p[0] / 1000.0 * w, p[1] / 1000.0 * h) for p in stroke]
            if len(pts) > 1:
                self.canvas.create_line(
                    *(v for xy in pts for v in xy),
                    fill=NAVY,
                    width=6,
                    smooth=True,
                    splinesteps=20,
                    capstyle="round",
                    joinstyle="round",
                )
            elif pts:
                x, y = pts[0]
                self.canvas.create_oval(
                    x - 4, y - 4, x + 4, y + 4, fill=NAVY, outline=""
                )

        # punto inicial (cian) y Punto final (coral)
        first_raw = self.strokes[0][0]
        last_raw = self.strokes[-1][-1]
        sx, sy = first_raw[0] / 1000.0 * w, first_raw[1] / 1000.0 * h
        ex, ey = last_raw[0] / 1000.0 * w, last_raw[1] / 1000.0 * h

        r = 7.0
        self.canvas.create_oval(
            sx - r, sy - r, sx + r, sy + r,
            fill=CYAN, outline="#ffffff", width=1
        )
        self.canvas.create_oval(
            ex - r, ey - r, ex + r, ey + r,
            fill=CORAL, outline="#ffffff", width=1
        )

    def _sphere_curve(self, points, fill: str, width: int = 1, dash=()):
        cx, cy, radius = self._sphere_dimensions()
        last = None
        for p in points:
            sx, sy, front = self.sphere.project(p, cx, cy, radius)
            if front > 0.003:
                if last is not None:
                    kw = {"dash": dash} if dash else {}
                    self.canvas.create_line(
                        *last, sx, sy, fill=fill, width=width, capstyle="round", **kw
                    )
                last = (sx, sy)
            else:
                last = None

    def _draw_3d_scene(self):
        cx, cy, r = self._sphere_dimensions()
        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill="#eef6f4", outline=TEAL, width=2
        )
        for phi_deg in range(-75, 90, 15):
            phi = math.radians(phi_deg)
            self._sphere_curve(
                [sphere_from_angles(math.radians(i), phi) for i in range(-180, 181, 4)],
                fill="#d3e3de",
                width=1,
            )
        for theta_deg in range(-180, 180, 15):
            theta = math.radians(theta_deg)
            self._sphere_curve(
                [sphere_from_angles(theta, math.radians(i)) for i in range(-89, 90, 4)],
                fill="#d3e3de",
                width=1,
            )

        if len(self.future) > 1:
            self._sphere_curve(self.future, ESTIMATED, width=3, dash=(5, 6))

        for stroke in self.strokes:
            coords = [p[:3] for p in stroke]
            if len(coords) > 1:
                self._sphere_curve(coords, NAVY, width=5)
            elif coords:
                sx, sy, z = self.sphere.project(coords[0], cx, cy, r)
                if z >= 0:
                    self.canvas.create_oval(
                        sx - 4, sy - 4, sx + 4, sy + 4, fill=NAVY, outline=""
                    )

        if self.strokes:
            sx, sy, sz = self.sphere.project(self.strokes[0][0][:3], cx, cy, r)
            if sz >= 0:
                self.canvas.create_oval(
                    sx - 7, sy - 7, sx + 7, sy + 7,
                    fill=CYAN, outline="#ffffff", width=1
                )
            ex, ey, ez = self.sphere.project(self.strokes[-1][-1][:3], cx, cy, r)
            if ez >= 0:
                self.canvas.create_oval(
                    ex - 7, ey - 7, ex + 7, ey + 7,
                    fill=CORAL, outline="#ffffff", width=1
                )
        else:
            self.canvas.create_text(
                cx,
                cy + r + 22,
                text="Dibuja sobre la superficie de la esfera (clic derecho para rotar)",
                fill=MUTED,
                font=("Segoe UI", 10),
            )

    def redraw(self):
        if self.canvas is None or not self.canvas.winfo_exists():
            return
        self.canvas.delete("all")
        if self.mode == "2d":
            self._draw_2d_scene()
        else:
            self._draw_3d_scene()

        if self.status_label:
            n_strokes = len(self.strokes)
            n_pts = sum(len(s) for s in self.strokes)
            if len(self._mode_info) != 27:
                self.status_label.configure(
                    text=f"Modelos {self.mode.upper()}: {len(self._mode_info)}/27"
                )
            else:
                self.status_label.configure(
                    text=f"{n_strokes} trazos · {n_pts} pts"
                )

    def run(self):
        if self.warnings:
            print("Avisos al cargar modelos:", *self.warnings, sep="\n")
        self.root.mainloop()


def main():
    HMMApp().run()


if __name__ == "__main__":
    main()