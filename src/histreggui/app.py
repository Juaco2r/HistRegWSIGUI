from __future__ import annotations

import inspect
import json
import os
import shutil
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


# -----------------------------------------------------------------------------
# Frozen-safe paths and imports
# -----------------------------------------------------------------------------
def runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2]


ROOT = runtime_root()
SRC_ROOT = Path(__file__).resolve().parents[1]
if not getattr(sys, "frozen", False) and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# DeeperHistReg contains a few imports that expect its package directory to be
# directly importable. The build script therefore includes the installed source
# tree as runtime data, and this path makes those imports work when frozen.
for candidate in (ROOT / "deeperhistreg", ROOT / "_internal" / "deeperhistreg"):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Backward compatibility for repositories that still vendor a libvips folder.
_DLL_HANDLES: list[object] = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for candidate in (
        ROOT / "external" / "libvips" / "vips-dev-8.18" / "bin",
        ROOT / "external" / "libvips" / "bin",
    ):
        if candidate.exists():
            _DLL_HANDLES.append(os.add_dll_directory(str(candidate)))
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")

import torch  # noqa: E402
import deeperhistreg  # noqa: E402

from histreggui.hardware import (  # noqa: E402
    CUDAInfo,
    configure_registration_device,
    detect_cuda,
    format_cuda_summary,
)


# -----------------------------------------------------------------------------
# Configuration and helpers
# -----------------------------------------------------------------------------
def load_build_info() -> dict[str, str]:
    candidates = (
        ROOT / "histreggui" / "build_info.json",
        Path(__file__).with_name("build_info.json"),
    )
    for path in candidates:
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
    return {
        "variant": "unknown",
        "platform": sys.platform,
        "architecture": "unknown",
        "torch_variant": "unknown",
    }


BUILD_INFO = load_build_info()


def build_presets() -> dict[str, object]:
    """Discover no-argument DeeperHistReg configuration factories."""

    presets: dict[str, object] = {}
    cfg = deeperhistreg.configs

    for name in dir(cfg):
        if name.startswith("_"):
            continue
        obj = getattr(cfg, name)
        if not callable(obj):
            continue

        try:
            if len(inspect.signature(obj).parameters) != 0:
                continue
        except Exception:
            continue

        try:
            value = obj()
            if isinstance(value, dict):
                presets[name.replace("_", " ")] = obj
        except Exception:
            pass

    preferred = [
        "default_initial_nonrigid",
        "default_initial_nonrigid_fast",
        "default_nonrigid",
        "default_nonrigid_fast",
        "default_initial",
    ]

    def sort_key(item: tuple[str, object]) -> tuple[int, object]:
        function_name = getattr(item[1], "__name__", "")
        if function_name in preferred:
            return (0, preferred.index(function_name))
        return (1, item[0].lower())

    return dict(sorted(presets.items(), key=sort_key))


PRESETS = build_presets()
if not PRESETS:
    PRESETS = {"default initial nonrigid": deeperhistreg.configs.default_initial_nonrigid}


def load_preview(path: Path, max_side: int = 600) -> Image.Image:
    """Load a downsampled RGB preview without changing registration inputs."""

    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = max(width, height) / max_side if max(width, height) > max_side else 1.0
    size = (max(1, int(width / scale)), max(1, int(height / scale)))
    return image.resize(size)


def pick_loader_for_warp(_path: Path):
    # The portable application intentionally uses PIL for the final warp.
    return deeperhistreg.loaders.PILLoader


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Histology Registration (DeeperHistReg)")
        self.geometry("1240x820")
        self.minsize(980, 650)

        self.fixed_path: Path | None = None
        self.moving_path: Path | None = None

        self.result_preview_imgtk: ImageTk.PhotoImage | None = None
        self.fixed_preview_imgtk: ImageTk.PhotoImage | None = None
        self.moving_preview_imgtk: ImageTk.PhotoImage | None = None

        self.preset_var = tk.StringVar(value=next(iter(PRESETS.keys())))
        self.save_intermediate_var = tk.BooleanVar(value=False)
        self.use_cuda_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready.")
        self.cuda_status_var = tk.StringVar(value="Checking hardware...")

        # Fast initial inspection; the explicit menu action performs a real
        # allocation probe.
        self.cuda_info: CUDAInfo = detect_cuda(torch, probe=False)

        self._build_menu()
        self._build_ui()
        self._update_cuda_controls()

    # ------------------------------------------------------------------ UI --
    def _build_menu(self) -> None:
        menu_bar = tk.Menu(self)
        hardware_menu = tk.Menu(menu_bar, tearoff=False)
        hardware_menu.add_command(
            label="Check CUDA availability...",
            command=lambda: self.check_cuda(show_dialog=True),
        )
        hardware_menu.add_checkbutton(
            label="Use CUDA acceleration when available",
            variable=self.use_cuda_var,
            command=self._cuda_toggled,
        )
        self.cuda_menu_index = int(hardware_menu.index("end"))
        hardware_menu.add_separator()
        hardware_menu.add_command(label="Build information...", command=self.show_build_info)
        menu_bar.add_cascade(label="Hardware", menu=hardware_menu)
        self.config(menu=menu_bar)
        self.hardware_menu = hardware_menu

    def _build_ui(self) -> None:
        top = ttk.LabelFrame(self, text="Registration settings", padding=12)
        top.pack(fill="x", padx=10, pady=(10, 4))
        top.columnconfigure(2, weight=1)

        ttk.Label(top, text="Target (Fixed) image:").grid(row=0, column=0, sticky="w")
        ttk.Button(top, text="Select...", command=self.select_fixed).grid(
            row=0, column=1, padx=8, pady=3
        )
        self.fixed_lbl = ttk.Label(top, text="(none)")
        self.fixed_lbl.grid(row=0, column=2, sticky="w")

        ttk.Label(top, text="Moving (Warp) image:").grid(row=1, column=0, sticky="w")
        ttk.Button(top, text="Select...", command=self.select_moving).grid(
            row=1, column=1, padx=8, pady=3
        )
        self.moving_lbl = ttk.Label(top, text="(none)")
        self.moving_lbl.grid(row=1, column=2, sticky="w")

        ttk.Label(top, text="Registration preset:").grid(row=2, column=0, sticky="w")
        preset = ttk.Combobox(
            top,
            textvariable=self.preset_var,
            values=list(PRESETS.keys()),
            state="readonly",
            width=38,
        )
        preset.grid(row=2, column=1, padx=8, pady=3, sticky="w")

        ttk.Checkbutton(
            top,
            text="Save intermediate results",
            variable=self.save_intermediate_var,
        ).grid(row=2, column=2, sticky="w")

        self.cuda_check = ttk.Checkbutton(
            top,
            text="Use CUDA acceleration (NVIDIA)",
            variable=self.use_cuda_var,
            command=self._cuda_toggled,
        )
        self.cuda_check.grid(row=3, column=1, padx=8, pady=(7, 3), sticky="w")

        ttk.Label(top, textvariable=self.cuda_status_var).grid(
            row=3, column=2, sticky="w", padx=(4, 0)
        )

        self.run_button = ttk.Button(top, text="Run registration", command=self.run_clicked)
        self.run_button.grid(row=4, column=1, pady=(10, 2), sticky="w")

        previews = ttk.Frame(self, padding=10)
        previews.pack(fill="both", expand=True)

        self.fixed_canvas = tk.Label(
            previews, text="Fixed preview", relief="groove", width=40, anchor="center"
        )
        self.moving_canvas = tk.Label(
            previews, text="Moving preview", relief="groove", width=40, anchor="center"
        )
        self.result_canvas = tk.Label(
            previews, text="Result preview", relief="groove", width=40, anchor="center"
        )

        self.fixed_canvas.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        self.moving_canvas.grid(row=0, column=1, padx=6, pady=6, sticky="nsew")
        self.result_canvas.grid(row=0, column=2, padx=6, pady=6, sticky="nsew")

        for column in range(3):
            previews.columnconfigure(column, weight=1)
        previews.rowconfigure(0, weight=1)

        bottom = ttk.Frame(self, padding=(12, 6, 12, 10))
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status_var).pack(anchor="w")

    def _ui(self, function, *args, **kwargs) -> None:
        self.after(0, lambda: function(*args, **kwargs))

    def _set_status(self, text: str) -> None:
        self._ui(self.status_var.set, text)

    def _show_error(self, title: str, message: str) -> None:
        self._ui(messagebox.showerror, title, message)

    # ------------------------------------------------------------ Hardware --
    def _update_cuda_controls(self) -> None:
        self.cuda_status_var.set(format_cuda_summary(self.cuda_info))
        state = "normal" if self.cuda_info.available else "disabled"
        self.hardware_menu.entryconfigure(self.cuda_menu_index, state=state)
        if self.cuda_info.available:
            self.cuda_check.state(["!disabled"])
        else:
            self.use_cuda_var.set(False)
            self.cuda_check.state(["disabled"])

    def _cuda_toggled(self) -> None:
        if self.use_cuda_var.get() and not self.cuda_info.available:
            self.use_cuda_var.set(False)
            messagebox.showwarning(
                "CUDA unavailable",
                "CUDA cannot be enabled in the current application/runtime.\n\n"
                + self.cuda_info.reason,
            )

    def check_cuda(self, *, show_dialog: bool) -> None:
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            self.cuda_info = detect_cuda(torch, probe=True)
            self._update_cuda_controls()
        finally:
            self.config(cursor="")

        if show_dialog:
            if self.cuda_info.available:
                details = (
                    f"CUDA is available.\n\n"
                    f"PyTorch CUDA runtime: {self.cuda_info.torch_cuda_version}\n"
                    f"Detected GPU(s):\n- " + "\n- ".join(self.cuda_info.device_names)
                )
                messagebox.showinfo("CUDA availability", details)
            else:
                messagebox.showwarning("CUDA availability", self.cuda_info.reason)

    def show_build_info(self) -> None:
        text = (
            f"Build variant: {BUILD_INFO.get('variant', 'unknown')}\n"
            f"Platform: {BUILD_INFO.get('platform', 'unknown')}\n"
            f"Architecture: {BUILD_INFO.get('architecture', 'unknown')}\n"
            f"PyTorch package: {BUILD_INFO.get('torch_variant', 'unknown')}\n"
            f"PyTorch version: {torch.__version__}\n"
            f"CUDA status: {format_cuda_summary(self.cuda_info)}"
        )
        messagebox.showinfo("Build information", text)

    # --------------------------------------------------------------- Inputs --
    def select_fixed(self) -> None:
        path = filedialog.askopenfilename(
            title="Select fixed (target) image",
            filetypes=[
                ("Images", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.fixed_path = Path(path)
        self.fixed_lbl.config(text=str(self.fixed_path))
        self._update_preview(self.fixed_path, which="fixed")

    def select_moving(self) -> None:
        path = filedialog.askopenfilename(
            title="Select moving (warp) image",
            filetypes=[
                ("Images", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.moving_path = Path(path)
        self.moving_lbl.config(text=str(self.moving_path))
        self._update_preview(self.moving_path, which="moving")

    def _update_preview(self, path: Path, which: str) -> None:
        try:
            image = load_preview(path, max_side=420)
            image_tk = ImageTk.PhotoImage(image)
            if which == "fixed":
                self.fixed_preview_imgtk = image_tk
                self.fixed_canvas.config(image=image_tk, text="")
            else:
                self.moving_preview_imgtk = image_tk
                self.moving_canvas.config(image=image_tk, text="")
        except Exception as exc:
            messagebox.showerror("Preview error", str(exc))

    def _set_result_preview(self, image: Image.Image) -> None:
        image_tk = ImageTk.PhotoImage(image)
        self.result_preview_imgtk = image_tk
        self.result_canvas.config(image=image_tk, text="")

    # --------------------------------------------------------- Registration --
    def run_clicked(self) -> None:
        if not self.fixed_path or not self.moving_path:
            messagebox.showwarning("Missing input", "Please select both fixed and moving images.")
            return

        use_cuda = bool(self.use_cuda_var.get())
        if use_cuda:
            # Recheck immediately before work starts, because a driver/device can
            # disappear after the application was opened.
            self.cuda_info = detect_cuda(torch, probe=True)
            self._update_cuda_controls()
            if not self.cuda_info.available:
                messagebox.showwarning(
                    "CUDA unavailable",
                    "CUDA was selected but is not currently usable. The registration will run on CPU.\n\n"
                    + self.cuda_info.reason,
                )
                use_cuda = False

        fixed = self.fixed_path
        moving = self.moving_path
        preset_key = self.preset_var.get()
        save_intermediate = bool(self.save_intermediate_var.get())

        self.run_button.config(state="disabled")
        threading.Thread(
            target=self._run_registration,
            args=(fixed, moving, preset_key, save_intermediate, use_cuda),
            daemon=True,
        ).start()

    def _run_registration(
        self,
        fixed: Path,
        moving: Path,
        preset_key: str,
        save_intermediate: bool,
        use_cuda: bool,
    ) -> None:
        log_path: Path | None = None
        try:
            output_directory = fixed.parent
            warped_output = output_directory / f"{moving.stem}_warped_to_{fixed.stem}.tif"
            run_directory = output_directory / f"Run_{timestamp()}"
            run_directory.mkdir(parents=True, exist_ok=True)

            preset_function = PRESETS[preset_key]
            function_name = getattr(preset_function, "__name__", preset_key)
            device = "cuda:0" if use_cuda else "cpu"

            self._set_status(f"Preparing parameters: {function_name} ...")
            parameters = configure_registration_device(preset_function(), device)  # type: ignore[operator]
            loader = pick_loader_for_warp(moving)

            if use_cuda:
                gpu_name = self.cuda_info.device_names[0]
                self._set_status(f"Running registration on CUDA: {gpu_name} ...")
            else:
                self._set_status("Running registration on CPU ...")

            registration = deeperhistreg.direct_registration.DeeperHistReg_FullResolution(
                registration_parameters=parameters
            )
            registration.run_registration(str(moving), str(fixed), str(run_directory))

            self._set_status("Finding displacement field ...")
            displacement_candidates = (
                list(run_directory.rglob("displacement_field.mha"))
                + list(run_directory.rglob("*disp*.mha"))
                + list(run_directory.rglob("*.mha"))
            )
            if not displacement_candidates:
                raise FileNotFoundError(
                    f"No displacement field was found under: {run_directory}"
                )
            displacement_field = next(
                (
                    path
                    for path in displacement_candidates
                    if path.name.lower() == "displacement_field.mha"
                ),
                displacement_candidates[0],
            )

            self._set_status("Warping moving image and saving TIFF ...")
            deeperhistreg.apply_deformation(
                source_image_path=str(moving),
                target_image_path=str(fixed),
                warped_image_path=str(warped_output),
                displacement_field_path=str(displacement_field),
                loader=loader,
                saver=deeperhistreg.savers.TIFFSaver,
                save_params=deeperhistreg.savers.tiff_saver.default_params,
                level=0,
                pad_value=255,
                save_source_only=True,
                to_template_shape=True,
                to_save_target_path=None,
            )

            try:
                result_image = load_preview(warped_output, max_side=420)
                self._ui(self._set_result_preview, result_image)
            except Exception:
                pass

            if not save_intermediate:
                shutil.rmtree(run_directory, ignore_errors=True)

            self._set_status(f"Done. Saved: {warped_output}")
            self._ui(
                messagebox.showinfo,
                "Done",
                f"Warped image saved:\n{warped_output}\n\nExecution device: {device}",
            )

        except Exception as exc:
            trace = traceback.format_exc()
            self._set_status("Error.")

            try:
                log_path = fixed.parent / "HistRegGUI_error.log"
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write("\n" + "=" * 80 + "\n")
                    handle.write(f"[ERROR] {datetime.now().isoformat()}\n")
                    handle.write(f"Fixed:  {fixed}\n")
                    handle.write(f"Moving: {moving}\n")
                    handle.write(f"CUDA requested: {use_cuda}\n")
                    handle.write(trace)
                    handle.write("\n")
            except Exception:
                log_path = None

            message = (
                "An error occurred during registration.\n\n"
                f"Error: {exc!r}\n\n"
                "Full traceback:\n"
                f"{trace}"
            )
            if log_path:
                message += f"\n\nLog saved to:\n{log_path}"
            self._show_error("Registration error", message)
        finally:
            self._ui(self.run_button.config, state="normal")


def _self_test_output_path(argv: list[str]) -> Path | None:
    """Return the optional path supplied after ``--self-test-output``."""

    try:
        index = argv.index("--self-test-output")
    except ValueError:
        return None

    if index + 1 >= len(argv):
        raise ValueError("--self-test-output requires a file path")
    return Path(argv[index + 1]).expanduser().resolve()


def run_self_test(output_path: Path | None = None) -> None:
    """Validate imports and resources without opening a Tk window.

    PyInstaller's ``--windowed`` mode may set ``sys.stdout`` and ``sys.stderr``
    to ``None`` on macOS. Writing the result to a file makes the smoke test
    reliable for both Intel and Apple Silicon application bundles.
    """

    if not PRESETS:
        raise RuntimeError("No DeeperHistReg registration presets were discovered.")

    info = detect_cuda(torch, probe=False)
    payload = {
        "status": "ok",
        "preset_count": len(PRESETS),
        "build": BUILD_INFO,
        "torch_version": str(torch.__version__),
        "cuda_compiled": info.compiled_with_cuda,
        "cuda_available": info.available,
    }
    serialized = json.dumps(payload, indent=2)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")

    if sys.stdout is not None:
        print(serialized)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        run_self_test(_self_test_output_path(sys.argv))
    else:
        Image.MAX_IMAGE_PIXELS = None
        App().mainloop()
