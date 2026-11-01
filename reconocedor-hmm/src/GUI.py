"""Aplicación de escritorio: menú, lienzo 2D, esfera 3D, clasificación y proyección.

Solo utiliza Tkinter (sugerido por el enunciado) y el núcleo HMM nativo.
No incluye modo ni botón de entrenamiento/calibración.

3D: proyección ortográfica real de puntos sobre esfera unidad, picking
(rayo-cámara ∩ esfera) y rotación de cámara. Mouse izquierdo dibuja sobre la
superficie; derecho rota; rueda acerca/aleja; botón reorienta.
"""
from __future__ import annotations
import math
import json
import statistics
import time
import ctypes
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import numpy as np
from src.FeatureExtractor import (LETTERS, extract_2d, extract_3d,
                                  normalize_sphere, sphere_from_angles)
from src.HMMEngine import HMM, classify, project_2d, project_3d

ROOT = Path(__file__).resolve().parents[1]
BG = '#0b1422'
PANEL = '#121f31'
STROKE = '#62d5d0'
FUTURE = '#ffca78'
MUTED = '#93a6c1'
TEXT = '#eff7ff'


def load_bundled_fonts():
    """Carga fuentes privadas en Windows sin instalación permanente."""
    if sys.platform != 'win32':
        return []
    loaded = []
    for filename in ('selawk.ttf', 'PlaywriteAR-Regular.ttf'):
        path = ROOT / 'assets' / 'fonts' / filename
        if path.is_file():
            try:
                if ctypes.windll.gdi32.AddFontResourceExW(str(path), 0x10, 0):
                    loaded.append(str(path))
            except (AttributeError, OSError):
                pass
    return loaded


class SphereCamera:
    def __init__(self):
        self.yaw, self.pitch, self.zoom = 0., 0., 1.

    def matrix(self) -> np.ndarray:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        return pitch @ yaw

    def project(self, p, cx, cy, radius):
        x, y, z = self.matrix() @ normalize_sphere(p)
        return cx + float(x) * radius, cy - float(y) * radius, float(z)

    def pick(self, x, y, cx, cy, radius):
        nx, ny = (x-cx)/radius, -(y-cy)/radius
        r2 = nx*nx + ny*ny
        if r2 > 1.:
            return None
        camera_p = np.array([nx, ny, math.sqrt(max(0.,1.-r2))])
        return (self.matrix().T @ camera_p).tolist()


class HMMApp:
    def __init__(self):
        self._private_fonts = load_bundled_fonts()
        self.root = tk.Tk()
        self.root.title('Unillanos · Reconocedor HMM 2D/3D')
        self.root.geometry('1250x790')
        self.root.minsize(960, 630)
        self.root.configure(bg=BG)
        self._style()
        self.models = {'2d': {}, '3d': {}}
        self.warnings = []
        for mode in self.models:
            for c in LETTERS:
                filename = ROOT / 'models' / mode / f'{c}.json'
                if filename.is_file():
                    try:
                        self.models[mode][c] = HMM.load(filename)
                    except Exception as exc:
                        self.warnings.append(f'{mode} {c}: {exc}')
        self.references = self._load_references()
        self.mode = ''
        self.canvas = None
        self.strokes = []
        self.prediction = []
        self.future = []
        self.sphere = SphereCamera()
        self.drag_rotate = None
        self._inference_id = None
        self.last_observations = []
        self.start_time = None
        self.label_status = None
        self.show_home()

    def _load_references(self):
        """Ejemplos vectoriales del JSON real, sin depender de tipografías del SO."""
        path = ROOT / 'data/raw/glifo-dataset-2026-10-31.json'
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            result = {}
            for char in LETTERS:
                candidates = [sample for sample in data.get(char, [])
                              if isinstance(sample, list) and sample
                              and all(isinstance(t, list) and t for t in sample)]
                if not candidates:
                    continue
                # Selección determinista: número de trazos modal y longitud mediana.
                counts = [len(sample) for sample in candidates]
                common = max(set(counts), key=lambda n: (counts.count(n), -n))
                candidates = [sample for sample in candidates if len(sample) == common]
                candidates.sort(key=lambda sample: sum(len(t) for t in sample))
                result[char] = candidates[len(candidates)//2]
            return result
        except (OSError, ValueError, TypeError, KeyError):
            return {}

    def _draw_reference(self, canvas, char, width=32, height=32):
        strokes = self.references.get(char)
        if not strokes:
            canvas.create_text(width/2, height/2, text=char, fill=TEXT,
                               font=('Segoe UI', 15))
            return
        xs = [p[0] for stroke in strokes for p in stroke]
        ys = [p[1] for stroke in strokes for p in stroke]
        left, top = min(xs), min(ys)
        span_x, span_y = max(xs)-left, max(ys)-top
        scale = min((width-8)/max(span_x, 1), (height-8)/max(span_y, 1))
        offset_x = (width-span_x*scale)/2
        offset_y = (height-span_y*scale)/2
        for stroke in strokes:
            coords = [(offset_x+(p[0]-left)*scale,
                       offset_y+(p[1]-top)*scale) for p in stroke]
            if len(coords) > 1:
                canvas.create_line(*(v for xy in coords for v in xy),
                                   fill=TEXT, width=2, capstyle='round', joinstyle='round')
            elif coords:
                x, y = coords[0]
                canvas.create_oval(x-2,y-2,x+2,y+2,fill=TEXT,outline='')

    @staticmethod
    def _letter_font(char, size=19):
        return ('Playwrite AR' if char in ('a', 'q') else 'Selawik', size, 'normal')

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('TFrame', background=BG)
        style.configure('Panel.TFrame', background=PANEL)
        style.configure('TLabel', background=BG, foreground=TEXT, font=('Segoe UI', 11))
        style.configure('Title.TLabel', background=BG, foreground=TEXT, font=('Segoe UI', 24, 'bold'))
        style.configure('Sub.TLabel', background=BG, foreground=MUTED, font=('Segoe UI', 10))
        style.configure('Panel.TLabel', background=PANEL, foreground=TEXT, font=('Segoe UI', 11))
        style.configure('PanelSub.TLabel', background=PANEL, foreground=MUTED, font=('Segoe UI', 10))
        style.configure('TButton', padding=10, font=('Segoe UI', 11), foreground=TEXT,
                        background='#24435c', borderwidth=0)
        style.map('TButton', background=[('active', '#2c657e')])

    def _clear_window(self):
        if self._inference_id:
            self.root.after_cancel(self._inference_id)
            self._inference_id = None
        for child in self.root.winfo_children():
            child.destroy()

    def show_home(self):
        self._unbind_ranking_scroll()
        self._clear_window()
        self.mode = ''
        shell = ttk.Frame(self.root, padding=35)
        shell.pack(fill='both', expand=True)
        ttk.Label(shell, text='UNIVERSIDAD DE LOS LLANOS · PROCESOS ESTOCÁSTICOS', style='Sub.TLabel').pack(anchor='w')
        ttk.Label(shell, text='Reconocedor de caracteres', style='Title.TLabel').pack(anchor='w', pady=(35,5))
        ttk.Label(shell, text='27 minúsculas · Modelos Ocultos de Markov implementados desde cero',
                  style='Sub.TLabel').pack(anchor='w', pady=(0,30))
        for mode, title, sub in (
            ('2d', 'Modo 2D — plano cartesiano', 'Trazo, clasificación probabilística y proyección de continuación.'),
            ('3d', 'Modo 3D — esfera', 'Trazo sobre una superficie esférica, rotación, zoom y autocompletado.'),
        ):
            frame = ttk.Frame(shell, style='Panel.TFrame', padding=20)
            frame.pack(fill='x', pady=8)
            ttk.Label(frame, text=title, font=('Segoe UI', 17, 'bold'),
                      style='Panel.TLabel').pack(anchor='w')
            ttk.Label(frame, text=sub, style='PanelSub.TLabel').pack(anchor='w', pady=8)
            ttk.Button(frame, text='Abrir '+mode.upper(), command=lambda m=mode: self.open_mode(m)).pack(anchor='e')
        ttk.Button(shell, text='Ayuda / cómo dibujar', command=self.show_help).pack(anchor='w', pady=22)
        ttk.Label(shell, text='Entrenamiento externo; la aplicación solo realiza inferencia en tiempo real.',
                  style='Sub.TLabel').pack(anchor='w')

    def open_mode(self, mode):
        self._clear_window()
        self.mode = mode
        self.strokes, self.prediction, self.future = [], [], []
        self.sphere = SphereCamera()
        self.start_time = time.monotonic()
        self._mode_info = self.models[mode]
        main = ttk.Frame(self.root, padding=18)
        main.pack(fill='both', expand=True)
        top = ttk.Frame(main)
        top.pack(fill='x', pady=(0,12))
        ttk.Button(top, text='← Menú', command=self.show_home).pack(side='left')
        ttk.Label(top, text='Reconocedor HMM · '+mode.upper(),
                  font=('Segoe UI', 17, 'bold')).pack(side='left', padx=16)
        ttk.Button(top, text='Ayuda', command=self.show_help).pack(side='right')
        if mode == '3d':
            ttk.Button(top, text='Reorientar esfera', command=self.reset_camera).pack(side='right', padx=10)
        body = ttk.Frame(main)
        body.pack(fill='both', expand=True)
        left = ttk.Frame(body, style='Panel.TFrame', padding=12)
        left.pack(side='left', fill='both', expand=True, padx=(0,12))
        right = ttk.Frame(body, style='Panel.TFrame', width=300, padding=12)
        right.pack(side='right', fill='y')
        right.pack_propagate(False)
        hint = ('Izquierdo: dibujar sobre esfera · Derecho: rotar · Rueda: zoom'
                if mode == '3d' else 'Mantén pulsado el ratón para dibujar · Suelta para levantar el lápiz')
        ttk.Label(left, text=hint, style='PanelSub.TLabel').pack(anchor='w', pady=(2,8))
        self.canvas = tk.Canvas(left, bg='#101f31', highlightbackground='#24435c', highlightthickness=1)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda _: self.redraw())
        self.canvas.bind('<ButtonPress-1>', self.mouse_start)
        self.canvas.bind('<B1-Motion>', self.mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self.mouse_finish)
        self.canvas.bind('<ButtonPress-3>', self.rotate_start)
        self.canvas.bind('<B3-Motion>', self.rotate_move)
        self.canvas.bind('<MouseWheel>', self.mouse_zoom)
        self.canvas.bind('<Button-4>', lambda e: self.zoom_by(1.08))
        self.canvas.bind('<Button-5>', lambda e: self.zoom_by(1/1.08))
        controls = ttk.Frame(left, style='Panel.TFrame')
        controls.pack(fill='x', pady=(10,0))
        ttk.Button(controls, text='Limpiar lienzo', command=self.clear_canvas).pack(side='left')
        ttk.Button(controls, text='Reiniciar captura', command=self.restart_capture).pack(side='left', padx=8)
        self.label_status = ttk.Label(controls, text='Sin observaciones', style='PanelSub.TLabel')
        self.label_status.pack(side='right')
        ttk.Label(right, text='CLASIFICACIÓN · P(letra | observaciones)',
                  style='Panel.TLabel', font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0,8))
        warning = ('Modelos 3D SINTÉTICOS (no entrenados con trazos 3D reales).'
                   if mode == '3d' and any(m.metadata.get('source') == 'PROYECCION_2D_A_ESFERA_SINTETICA'
                                          for m in self._mode_info.values()) else '')
        if len(self._mode_info) != 27:
            warning = f'Faltan modelos de {mode.upper()} ({len(self._mode_info)}/27). Ejecuta el script offline.'
        if warning:
            tk.Label(right, bg=PANEL, fg=FUTURE, wraplength=275, justify='left',
                     text=warning, font=('Segoe UI', 9)).pack(anchor='w', pady=(0,8))
        list_box = tk.Frame(right, bg=PANEL)
        list_box.pack(fill='both', expand=True)
        scroll = tk.Scrollbar(list_box, orient='vertical', width=13,
                              bg='#315574', troughcolor='#0b1422',
                              activebackground='#62d5d0', relief='flat', bd=0)
        scroll.pack(side='right', fill='y')
        list_canvas = tk.Canvas(list_box, bg=PANEL, highlightthickness=0,
                                yscrollcommand=scroll.set, borderwidth=0)
        list_canvas.pack(side='left', fill='both', expand=True)
        scroll.configure(command=list_canvas.yview)
        self._ranking_canvas = list_canvas
        self.rows_frame = tk.Frame(list_canvas, bg=PANEL)
        list_canvas.bind('<Enter>', lambda e: self._bind_ranking_scroll())
        list_canvas.bind('<Leave>', lambda e: self._unbind_ranking_scroll())
        anchor = list_canvas.create_window((0,0), window=self.rows_frame, anchor='nw')
        self.rows_frame.bind('<Configure>',
                             lambda e: list_canvas.configure(scrollregion=list_canvas.bbox('all')))
        list_canvas.bind('<Configure>',
                         lambda e: list_canvas.itemconfigure(anchor, width=e.width))
        self.rows = []
        for _ in LETTERS:
            item = tk.Frame(self.rows_frame, bg=PANEL)
            item.pack(fill='x', pady=1)
            # Icono de glifo estático según referencia: tipografía visible en el panel.
            glyph = tk.Canvas(item, width=46, height=52, bg='#21394c', highlightthickness=0)
            glyph.pack(side='left', padx=(0,8))
            text = tk.Label(item, text='a', font=('Segoe UI', 11), bg=PANEL, fg=TEXT, anchor='w')
            text.pack(side='left', fill='x', expand=True)
            score = tk.Label(item, text='3.70%', font=('Consolas', 10),
                             bg=PANEL, fg=MUTED)
            score.pack(side='right')
            self.rows.append((glyph, text, score))
        self._update_rows([])
        self.redraw()

    def show_help(self):
        popup = tk.Toplevel(self.root)
        popup.configure(bg=BG)
        popup.title('Ayuda · Referencia de escritura')
        # Tamaño inicial calculado después de construir todos los controles.
        popup.transient(self.root)
        tk.Label(popup, bg=BG, fg=TEXT, text='Cómo dibujar las 27 letras',
                 font=('Segoe UI', 18, 'bold')).pack(pady=(18,12))
        guide = ('Dibuja las letras MINÚSCULAS como en las referencias.\n'
                 'a: círculo y rayita; q: cruz inferior.\n'
                 'i, j, ñ: levanta el lápiz para sus puntos/tilde;\n'
                 'no unas segmentos separados con líneas invisibles.\n\n'
                 'Modo 2D: dibuja con botón izquierdo.\n'
                 'Modo 3D: izquierdo sobre esfera para dibujar, derecho para girar,\n'
                 'rueda para zoom y botón Reorientar para volver a vista inicial.\n\n'
                 'El listado muestra probabilidades posteriores (priors iguales).\n'
                 'La línea punteada es una estimación, no una garantía.')
        tk.Label(popup, text=guide, bg=BG, fg=MUTED, justify='left',
                 font=('Segoe UI', 11)).pack(padx=18, anchor='w')
        letters = tk.Frame(popup, bg=BG)
        letters.pack(padx=18, pady=14)
        for i, letter in enumerate(LETTERS):
            # Corrección EXCLUSIVA de la ayuda: reservar espacio para los
            # descendentes de Playwrite AR, sin mover el texto según bbox().
            # Las casillas de clasificación permanecen idénticas a v0.7.
            preview = tk.Canvas(letters, width=52, height=64,
                                bg='#21394c', highlightthickness=0)
            preview.grid(row=i//9, column=i%9, padx=2, pady=2)
            preview.create_text(26, 25, text=letter, fill=TEXT,
                                anchor='center', font=self._letter_font(letter, 23))
        popup.update_idletasks()
        width = max(630, popup.winfo_reqwidth() + 24)
        height = max(540, popup.winfo_reqheight() + 24)
        screen_w, screen_h = popup.winfo_screenwidth(), popup.winfo_screenheight()
        width = min(width, screen_w - 60)
        height = min(height, screen_h - 90)
        popup.geometry(f'{width}x{height}')
        popup.minsize(width, height)

    def clear_canvas(self):
        if self._inference_id is not None:
            self.root.after_cancel(self._inference_id)
            self._inference_id = None
        self.strokes = []
        self.future = []
        self.prediction = []
        self.last_observations = []
        self._update_rows([])
        self.redraw()

    def restart_capture(self):
        self.clear_canvas()
        self.start_time = time.monotonic()
        if self.mode == '3d':
            self.sphere = SphereCamera()
        self.redraw()

    def reset_camera(self):
        self.sphere = SphereCamera()
        self.redraw()

    def _sphere_dimensions(self):
        w, h = max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)
        return w/2, h/2, min(w,h)*.40*self.sphere.zoom

    def _point_from_event(self, event):
        if self.mode == '2d':
            w, h = max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)
            x, y = (max(0,min(w,event.x))/w*1000,
                    max(0,min(h,event.y))/h*1000)
            return [x, y, (time.monotonic()-self.start_time)*1000]
        cx, cy, r = self._sphere_dimensions()
        position = self.sphere.pick(event.x,event.y,cx,cy,r)
        if position is None:
            return None
        return position + [(time.monotonic()-self.start_time)*1000]

    def mouse_start(self, event):
        p = self._point_from_event(event)
        if p is None:
            return
        self.strokes.append([p])
        self.redraw()

    def mouse_move(self, event):
        if not self.strokes:
            return
        p = self._point_from_event(event)
        if p is None:
            return
        last = self.strokes[-1][-1]
        d = math.dist(p[:2],last[:2]) if self.mode=='2d' else math.dist(p[:3],last[:3])*500
        if d < (3 if self.mode=='2d' else 3):
            return
        self.strokes[-1].append(p)
        self.redraw()
        self.schedule_inference()

    def mouse_finish(self, event):
        self.schedule_inference()

    def rotate_start(self, event):
        self.drag_rotate = (event.x, event.y)

    def rotate_move(self, event):
        if self.mode != '3d' or self.drag_rotate is None:
            return
        dx, dy = event.x-self.drag_rotate[0], event.y-self.drag_rotate[1]
        self.sphere.yaw += dx*.009
        self.sphere.pitch = max(-1.45,min(1.45,self.sphere.pitch+dy*.009))
        self.drag_rotate = (event.x,event.y)
        self.redraw()

    def zoom_by(self, factor):
        self.sphere.zoom = min(1.18,max(.55,self.sphere.zoom*factor))
        self.redraw()

    def mouse_zoom(self,event):
        if self.mode=='3d':
            self.zoom_by(1.08 if event.delta>0 else 1/1.08)

    def schedule_inference(self):
        if self._inference_id is None:
            self._inference_id = self.root.after(90, self.run_inference)

    def run_inference(self):
        self._inference_id = None
        if not self.strokes or len(self._mode_info) != 27:
            return
        try:
            if self.mode == '2d':
                obs = extract_2d(self.strokes, sectors=8, step_units=10.)
            else:
                obs = extract_3d(self.strokes, sectors=8, step_units=.018)
            self.last_observations = obs
            ranking = classify(self._mode_info, obs)
            self.prediction = ranking
            self._update_rows(ranking)
            best_char = ranking[0][0] if ranking else None
            model = self._mode_info.get(best_char)
            if model:
                current = self.strokes[-1][-1]
                self.future = (project_2d(model, obs, current[:2], sectors=8)
                               if self.mode == '2d' else
                               project_3d(model, obs, np.array(current[:3]), sectors=8))
            else:
                self.future = []
        except Exception as exc:
            self.future = []
            if self.label_status:
                self.label_status.configure(text=f'Error: {str(exc)[:55]}')
        self.redraw()

    def _bind_ranking_scroll(self):
        self.root.bind_all('<MouseWheel>', self._ranking_scroll)
        self.root.bind_all('<Button-4>', self._ranking_scroll)
        self.root.bind_all('<Button-5>', self._ranking_scroll)

    def _unbind_ranking_scroll(self):
        self.root.unbind_all('<MouseWheel>')
        self.root.unbind_all('<Button-4>')
        self.root.unbind_all('<Button-5>')

    def _ranking_scroll(self, event):
        if not hasattr(self, '_ranking_canvas') or not self._ranking_canvas.winfo_exists():
            return
        direction = (-1 if getattr(event, 'num', None) == 4 else
                     1 if getattr(event, 'num', None) == 5 else
                     -1 if event.delta > 0 else 1)
        self._ranking_canvas.yview_scroll(direction * 3, 'units')

    def _update_rows(self, ranking):
        ordered = ranking or [(char, 1/27, 0.) for char in LETTERS]
        for i, ((glyph, name, probability), (char, p, _)) in enumerate(zip(self.rows, ordered)):
            glyph.delete('all')
            # Tk Canvas posiciona el texto por su caja tipográfica, que en
            # Playwrite AR deja mucho espacio invisible arriba. Ajustar usando
            # el bbox real del glifo evita recortar 'a', 'q' y descendentes.
            item_id = glyph.create_text(23, 26, text=char, fill=TEXT, anchor='center',
                                        font=self._letter_font(char, 19))
            bounds = glyph.bbox(item_id)
            if bounds:
                x0, y0, x1, y1 = bounds
                glyph.move(item_id, 23 - (x0 + x1) / 2, 26 - (y0 + y1) / 2)
            name.configure(text=char, font=self._letter_font(char, 14),
                           fg=(STROKE if i==0 and ranking else TEXT))
            probability.configure(text=f'{100*p:6.2f} %', fg=(STROKE if i==0 and ranking else MUTED))

    def _canvas_2d(self):
        w, h = self.canvas.winfo_width(),self.canvas.winfo_height()
        for ratio, description in ((.16,'ascendente'),(.38,'altura x'),(.72,'línea base'),(.91,'descendente')):
            y = h*ratio
            self.canvas.create_line(20,y,w-20,y,fill='#2e465d',dash=(4,6))
            self.canvas.create_text(25,y-12,text=description,anchor='w',fill=MUTED,
                                    font=('Segoe UI',9))
        for stroke in self.strokes:
            pts = [(p[0]/1000*w,p[1]/1000*h) for p in stroke]
            if len(pts)>1:
                self.canvas.create_line(*(z for p in pts for z in p), fill=STROKE,
                                        width=4, capstyle='round', joinstyle='round',smooth=False)
            else:
                x,y=pts[0]
                self.canvas.create_oval(x-3,y-3,x+3,y+3,fill=STROKE,outline='')
        if len(self.future)>1:
            pts = [(x/1000*w,y/1000*h) for x,y in self.future]
            self.canvas.create_line(*(z for p in pts for z in p), fill=FUTURE,
                                    width=3, dash=(6,7), capstyle='round')

    def _sphere_curve(self, points, fill, width=1, dash=()):
        cx, cy, radius = self._sphere_dimensions()
        last = None
        for p in points:
            sx, sy, front = self.sphere.project(p,cx,cy,radius)
            if front > 0.003:
                if last is not None:
                    kw = {'dash': dash} if dash else {}
                    self.canvas.create_line(*last,sx,sy,fill=fill,width=width,**kw)
                last = (sx,sy)
            else:
                last = None

    def _canvas_3d(self):
        cx,cy,r=self._sphere_dimensions()
        self.canvas.create_oval(cx-r,cy-r,cx+r,cy+r,fill='#15334a',outline='#5b8ca4',width=2)
        # Mallado de latitudes y longitudes, visible solamente en hemisferio frontal.
        for phi_deg in range(-75,90,15):
            phi=math.radians(phi_deg)
            self._sphere_curve([sphere_from_angles(math.radians(i),phi)
                                for i in range(-180,181,3)],fill='#2b536d')
        for theta_deg in range(-180,180,15):
            theta=math.radians(theta_deg)
            self._sphere_curve([sphere_from_angles(theta,math.radians(i))
                                for i in range(-89,90,3)],fill='#2b536d')
        for stroke in self.strokes:
            coords=[p[:3] for p in stroke]
            if len(coords)>1:
                self._sphere_curve(coords,STROKE,width=4)
            elif coords:
                sx,sy,z=self.sphere.project(coords[0],cx,cy,r)
                if z>=0:
                    self.canvas.create_oval(sx-3,sy-3,sx+3,sy+3,fill=STROKE,outline='')
        if len(self.future)>1:
            self._sphere_curve(self.future,FUTURE,width=3,dash=(5,7))

    def redraw(self):
        if self.canvas is None or not self.canvas.winfo_exists() or not self.mode:
            return
        self.canvas.delete('all')
        if self.mode == '2d':
            self._canvas_2d()
        else:
            self._canvas_3d()
        if self.label_status:
            self.label_status.configure(text=f'{len(self.strokes)} trazos · '
                                         f'{sum(len(s) for s in self.strokes)} puntos · '
                                         f'{len(self.last_observations)} obs.')

    def run(self):
        if self.warnings:
            print('Modelos no cargados:', *self.warnings,sep='\n')
        self.root.mainloop()


def main():
    HMMApp().run()


if __name__ == '__main__':
    main()
