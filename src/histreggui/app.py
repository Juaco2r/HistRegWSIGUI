import os
import sys
import threading
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import traceback


from PIL import Image, ImageTk

# -----------------------------
# Frozen-safe root + deeperhistreg import fix
# -----------------------------
def get_root() -> Path:
    # When frozen, PyInstaller extracts files to sys._MEIPASS
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]

ROOT = get_root()

# DeeperHistReg internally imports top-level modules like "dhr_preprocessing".
# We bundle deeperhistreg as a real folder and add it to sys.path so those imports work.
dhr_dir = ROOT / "deeperhistreg"
if dhr_dir.exists():
    sys.path.insert(0, str(dhr_dir))

# -----------------------------
# DLL setup (portable)
# -----------------------------
VIPSBIN = ROOT / "external" / "libvips" / "vips-dev-8.18" / "bin"
if VIPSBIN.exists():
    os.add_dll_directory(str(VIPSBIN))
    os.environ["PATH"] = str(VIPSBIN) + os.pathsep + os.environ.get("PATH", "")

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

# Some parts of DeeperHistReg will try to use CUDA if params say "cuda"
torch.cuda.is_available = lambda: False  # type: ignore

# Hard guard: redirect Tensor.to("cuda") -> Tensor.to("cpu") to avoid CUDA init crashes
_old_to = torch.Tensor.to  # type: ignore[attr-defined]

def _to_cpu(self, *args, **kwargs):
    if len(args) >= 1:
        dev = args[0]
        if isinstance(dev, str) and dev.lower().startswith("cuda"):
            args = ("cpu",) + args[1:]
    if "device" in kwargs:
        dev = kwargs["device"]
        if isinstance(dev, str) and dev.lower().startswith("cuda"):
            kwargs["device"] = "cpu"
    return _old_to(self, *args, **kwargs)

torch.Tensor.to = _to_cpu  # type: ignore[attr-defined]

import deeperhistreg  # noqa: E402

import inspect

def build_presets():
    """
    Build dropdown presets from deeperhistreg.configs automatically.
    Keeps only callables that return a dict (registration parameters).
    """
    presets = {}

    cfg = deeperhistreg.configs
    for name in dir(cfg):
        if name.startswith("_"):
            continue
        obj = getattr(cfg, name)
        if callable(obj):
            # we only want config factories like default_initial_nonrigid()
            # ignore things that require args
            try:
                sig = inspect.signature(obj)
                if len(sig.parameters) != 0:
                    continue
            except Exception:
                continue

            # Try calling it; must return a dict
            try:
                val = obj()
                if isinstance(val, dict):
                    pretty = name.replace("_", " ")
                    presets[pretty] = obj
            except Exception:
                # ignore configs that cannot be called in this environment
                pass

    # Set default preference order (if present)
    preferred = [
        "default_initial_nonrigid",
        "default_nonrigid",
        "default_rigid",
        "default_initial",
    ]

    # Sort: preferred first, then alphabetical
    def sort_key(item):
        func_name = item[1].__name__
        if func_name in preferred:
            return (0, preferred.index(func_name))
        return (1, item[0].lower())

    presets = dict(sorted(presets.items(), key=sort_key))
    return presets

PRESETS = build_presets()

# Fallback if for some reason nothing is detected
if not PRESETS:
    PRESETS = {"default initial nonrigid": deeperhistreg.configs.default_initial_nonrigid}


# -----------------------------
# Helpers
# -----------------------------
TIFF_EXT = {".tif", ".tiff"}

def load_preview(path: Path, max_side=600) -> Image.Image:
    """Load a downsampled preview for GUI (does NOT affect registration)."""
    img = Image.open(path)
    img = img.convert("RGB")
    w, h = img.size
    scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
    new_size = (max(1, int(w / scale)), max(1, int(h / scale)))
    return img.resize(new_size)

def pick_loader_for_warp(path: Path):
    # V1 portable version: always use PIL loader
    return deeperhistreg.loaders.PILLoader


def windows_safe_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def force_device_cpu(obj):
    """Recursively replace device='cuda*' with 'cpu' inside nested dicts/lists."""
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if isinstance(v, str) and v.lower().startswith("cuda"):
                v = "cpu"
            if isinstance(k, str) and k.lower() == "device" and isinstance(v, str):
                if v.lower().startswith("cuda"):
                    v = "cpu"
            new[k] = force_device_cpu(v)
        return new
    if isinstance(obj, list):
        return [force_device_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(force_device_cpu(v) for v in obj)
    return obj




class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Histology Registration (DeeperHistReg)")
        self.geometry("1200x780")

        self.fixed_path: Path | None = None
        self.moving_path: Path | None = None

        self.result_preview_imgtk = None
        self.fixed_preview_imgtk = None
        self.moving_preview_imgtk = None

        default_key = next(iter(PRESETS.keys()))
        self.preset_var = tk.StringVar(value=default_key)

        self.save_intermediate_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready.")

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Target (Fixed) image:").grid(row=0, column=0, sticky="w")
        ttk.Button(top, text="Select...", command=self.select_fixed).grid(row=0, column=1, padx=8)
        self.fixed_lbl = ttk.Label(top, text="(none)")
        self.fixed_lbl.grid(row=0, column=2, sticky="w")

        ttk.Label(top, text="Moving (Warp) image:").grid(row=1, column=0, sticky="w")
        ttk.Button(top, text="Select...", command=self.select_moving).grid(row=1, column=1, padx=8)
        self.moving_lbl = ttk.Label(top, text="(none)")
        self.moving_lbl.grid(row=1, column=2, sticky="w")

        ttk.Label(top, text="Registration preset:").grid(row=2, column=0, sticky="w")
        preset = ttk.Combobox(
            top,
            textvariable=self.preset_var,
            values=list(PRESETS.keys()),
            state="readonly",
            width=35,
        )
        preset.grid(row=2, column=1, padx=8, sticky="w")

        ttk.Checkbutton(
            top,
            text="Save intermediate results",
            variable=self.save_intermediate_var,
        ).grid(row=2, column=2, sticky="w")

        ttk.Button(top, text="Run registration", command=self.run_clicked).grid(row=3, column=1, pady=10, sticky="w")

        mid = ttk.Frame(self, padding=10)
        mid.pack(fill="both", expand=True)

        self.fixed_canvas = tk.Label(mid, text="Fixed preview", relief="groove", width=50, anchor="center")
        self.moving_canvas = tk.Label(mid, text="Moving preview", relief="groove", width=50, anchor="center")
        self.result_canvas = tk.Label(mid, text="Result preview", relief="groove", width=50, anchor="center")

        self.fixed_canvas.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        self.moving_canvas.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")
        self.result_canvas.grid(row=0, column=2, padx=8, pady=8, sticky="nsew")

        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)
        mid.columnconfigure(2, weight=1)
        mid.rowconfigure(0, weight=1)

        bottom = ttk.Frame(self, padding=10)
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status_var).pack(anchor="w")

    def _ui(self, fn, *args, **kwargs):
        """Run a function on the Tk main thread (safe from worker threads)."""
        self.after(0, lambda: fn(*args, **kwargs))

    def _show_error(self, title: str, msg: str):
        self._ui(messagebox.showerror, title, msg)

    def select_fixed(self):
        path = filedialog.askopenfilename(
            title="Select fixed (target) image",
            filetypes=[("Images", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        self.fixed_path = Path(path)
        self.fixed_lbl.config(text=str(self.fixed_path))
        self._update_preview(self.fixed_path, which="fixed")

    def select_moving(self):
        path = filedialog.askopenfilename(
            title="Select moving (warp) image",
            filetypes=[("Images", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        self.moving_path = Path(path)
        self.moving_lbl.config(text=str(self.moving_path))
        self._update_preview(self.moving_path, which="moving")

    def _update_preview(self, path: Path, which: str):
        try:
            img = load_preview(path, max_side=420)
            imgtk = ImageTk.PhotoImage(img)
            if which == "fixed":
                self.fixed_preview_imgtk = imgtk
                self.fixed_canvas.config(image=imgtk, text="")
            elif which == "moving":
                self.moving_preview_imgtk = imgtk
                self.moving_canvas.config(image=imgtk, text="")
        except Exception as e:
            self._show_error("Preview error", str(e))

    def run_clicked(self):
        if not self.fixed_path or not self.moving_path:
            messagebox.showwarning("Missing input", "Please select both fixed and moving images.")
            return
        threading.Thread(target=self._run_registration, daemon=True).start()

    def _run_registration(self):
        try:
            fixed = self.fixed_path
            moving = self.moving_path
            assert fixed and moving

            out_dir = fixed.parent
            warped_out = out_dir / f"{moving.stem}_warped_to_{fixed.stem}.tif"

            stamp = windows_safe_stamp()
            run_dir = out_dir / f"Run_{stamp}"
            run_dir.mkdir(parents=True, exist_ok=True)

            self.status_var.set("Preparing parameters...")
            preset_fn = PRESETS[self.preset_var.get()]
            self.status_var.set(f"Preparing parameters: {preset_fn.__name__} ...")

            params = force_device_cpu(preset_fn())

            save_intermediate = self.save_intermediate_var.get()

            loader = pick_loader_for_warp(moving)

            self._ui(self.status_var.set, "Running registration (CPU)...")
            reg = deeperhistreg.direct_registration.DeeperHistReg_FullResolution(registration_parameters=params)
            reg.run_registration(str(moving), str(fixed), str(run_dir))

            self.status_var.set("Finding displacement field...")
            disp_candidates = (
                list(run_dir.rglob("displacement_field.mha"))
                + list(run_dir.rglob("*disp*.mha"))
                + list(run_dir.rglob("*.mha"))
            )
            if not disp_candidates:
                raise FileNotFoundError(f"No displacement field found under: {run_dir}")
            disp_field = next((p for p in disp_candidates if p.name.lower() == "displacement_field.mha"), disp_candidates[0])

            self.status_var.set("Warping moving image to target and saving...")
            saver = deeperhistreg.savers.TIFFSaver
            save_params = deeperhistreg.savers.tiff_saver.default_params

            deeperhistreg.apply_deformation(
                source_image_path=str(moving),
                target_image_path=str(fixed),
                warped_image_path=str(warped_out),
                displacement_field_path=str(disp_field),
                loader=loader,
                saver=saver,
                save_params=save_params,
                level=0,
                pad_value=255,
                save_source_only=True,
                to_template_shape=True,
                to_save_target_path=None,
            )

            try:
                img = load_preview(warped_out, max_side=420)
                imgtk = ImageTk.PhotoImage(img)
                self.result_preview_imgtk = imgtk
                self.result_canvas.config(image=imgtk, text="")
            except Exception:
                pass

            if not save_intermediate:
                try:
                    for p in sorted(run_dir.rglob("*"), reverse=True):
                        if p.is_file():
                            p.unlink()
                        else:
                            p.rmdir()
                    run_dir.rmdir()
                except Exception:
                    pass

            self.status_var.set(f"Done. Saved: {warped_out}")
            self._ui(messagebox.showinfo, "Done", f"Warped image saved:\n{warped_out}")

        except Exception as e:
            tb = traceback.format_exc()
            self.status_var.set("Error.")

            # Guardar log en la carpeta de salida (si ya existe fixed)
            try:
                out_dir = self.fixed_path.parent if self.fixed_path else Path.cwd()
                log_path = out_dir / "HistRegGUI_error.log"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"[ERROR] {datetime.now().isoformat()}\n")
                    f.write(f"Fixed:  {self.fixed_path}\n")
                    f.write(f"Moving: {self.moving_path}\n")
                    f.write(tb)
                    f.write("\n")
            except Exception:
                log_path = None

            msg = (
                "Ocurrió un error durante el registro.\n\n"
                f"Error: {repr(e)}\n\n"
                "Traceback completo:\n"
                f"{tb}"
            )
            if log_path:
                msg += f"\n\nLog guardado en:\n{log_path}"

            # Importante: mostrar popup desde el hilo principal
            self._show_error("Registration error", msg)



if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None
    app = App()
    app.mainloop()
