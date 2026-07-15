from __future__ import annotations

import gc
import inspect
import json
import os
import shutil
import sys
import threading
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Allow this file to be executed directly from the source tree as well as
# imported as a package or frozen by PyInstaller.
_EARLY_SRC_ROOT = Path(__file__).resolve().parents[1]
if not getattr(sys, "frozen", False) and str(_EARLY_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_EARLY_SRC_ROOT))

from histreggui import __version__
from histreggui.pillow_compat import install_pillow_tkinter_finder_alias

# Install the compatibility alias before importing ImageTk.  The packaged build
# also includes a dedicated PyInstaller hook for the underlying private module.
install_pillow_tkinter_finder_alias()
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

from histreggui.batch import (  # noqa: E402
    RegistrationBatchPlan,
    RegistrationPlanItem,
    build_registration_batch_plan,
    unique_paths,
    write_registration_manifest,
)
from histreggui.hardware import (  # noqa: E402
    CUDAInfo,
    configure_registration_device,
    detect_cuda,
    format_cuda_summary,
)
from histreggui.image_io import (  # noqa: E402
    LOADER_CHOICES,
    configure_registration_loader,
    deeperhistreg_loader_class,
    load_image_preview,
    resolve_loader_choice,
    supported_formats_text,
    tkinter_image_filetypes,
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
        "version": __version__,
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


def load_preview(path: Path, max_side: int = 600):
    """Load a memory-conscious preview using TIFF, WSI, or raster backends."""

    return load_image_preview(path, max_side=max_side)


def pick_loader_for_warp(loader_key: str):
    """Use the same loader for registration and final full-resolution warp."""

    return deeperhistreg_loader_class(deeperhistreg, loader_key)

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"Histology Registration (DeeperHistReg) v{__version__}")
        self.geometry("1280x900")
        self.minsize(1040, 720)

        self.fixed_path: Path | None = None
        self.moving_paths: list[Path] = []
        self.current_moving_path: Path | None = None
        self.moving_tree_paths: dict[str, Path] = {}
        self.moving_tree_iids: dict[str, str] = {}
        self.moving_statuses: dict[str, str] = {}
        self.moving_readers: dict[str, str] = {}

        self.result_preview_imgtk: ImageTk.PhotoImage | None = None
        self.fixed_preview_imgtk: ImageTk.PhotoImage | None = None
        self.moving_preview_imgtk: ImageTk.PhotoImage | None = None

        self.preset_var = tk.StringVar(value=next(iter(PRESETS.keys())))
        self.loader_var = tk.StringVar(value=next(iter(LOADER_CHOICES.keys())))
        self.loader_status_var = tk.StringVar(value="Input reader: automatic")
        self.moving_count_var = tk.StringVar(value="No moving images selected.")
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
        self.fixed_button = ttk.Button(top, text="Select...", command=self.select_fixed)
        self.fixed_button.grid(row=0, column=1, padx=8, pady=3, sticky="w")
        self.fixed_lbl = ttk.Label(top, text="(none)")
        self.fixed_lbl.grid(row=0, column=2, sticky="w")

        ttk.Label(top, text="Moving images:").grid(row=1, column=0, sticky="nw", pady=(4, 0))
        moving_buttons = ttk.Frame(top)
        moving_buttons.grid(row=1, column=1, padx=8, pady=3, sticky="nw")
        self.add_moving_button = ttk.Button(
            moving_buttons, text="Add images...", command=self.add_moving_images
        )
        self.add_moving_button.pack(side="left")
        self.remove_moving_button = ttk.Button(
            moving_buttons, text="Remove selected", command=self.remove_selected_moving
        )
        self.remove_moving_button.pack(side="left", padx=(6, 0))
        self.clear_moving_button = ttk.Button(
            moving_buttons, text="Clear", command=self.clear_moving_images
        )
        self.clear_moving_button.pack(side="left", padx=(6, 0))
        ttk.Label(top, textvariable=self.moving_count_var).grid(
            row=1, column=2, sticky="w", pady=(4, 0)
        )

        moving_frame = ttk.Frame(top)
        moving_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(5, 8))
        moving_frame.columnconfigure(0, weight=1)
        self.moving_tree = ttk.Treeview(
            moving_frame,
            columns=("reader", "status"),
            show="tree headings",
            height=5,
            selectmode="extended",
        )
        self.moving_tree.heading("#0", text="Moving image")
        self.moving_tree.heading("reader", text="Reader")
        self.moving_tree.heading("status", text="Status")
        self.moving_tree.column("#0", width=650, minwidth=260, stretch=True)
        self.moving_tree.column("reader", width=115, minwidth=90, stretch=False)
        self.moving_tree.column("status", width=220, minwidth=140, stretch=True)
        moving_scroll = ttk.Scrollbar(
            moving_frame, orient="vertical", command=self.moving_tree.yview
        )
        self.moving_tree.configure(yscrollcommand=moving_scroll.set)
        self.moving_tree.grid(row=0, column=0, sticky="nsew")
        moving_scroll.grid(row=0, column=1, sticky="ns")
        self.moving_tree.bind("<<TreeviewSelect>>", self._moving_selection_changed)

        ttk.Label(top, text="Registration preset:").grid(row=3, column=0, sticky="w")
        preset = ttk.Combobox(
            top,
            textvariable=self.preset_var,
            values=list(PRESETS.keys()),
            state="readonly",
            width=38,
        )
        preset.grid(row=3, column=1, padx=8, pady=3, sticky="w")

        ttk.Checkbutton(
            top,
            text="Save intermediate results",
            variable=self.save_intermediate_var,
        ).grid(row=3, column=2, sticky="w")

        ttk.Label(top, text="Input reader:").grid(row=4, column=0, sticky="w")
        loader_combo = ttk.Combobox(
            top,
            textvariable=self.loader_var,
            values=list(LOADER_CHOICES.keys()),
            state="readonly",
            width=38,
        )
        loader_combo.grid(row=4, column=1, padx=8, pady=3, sticky="w")
        loader_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_loader_status())
        ttk.Label(top, textvariable=self.loader_status_var).grid(
            row=4, column=2, sticky="w", padx=(4, 0)
        )

        self.cuda_check = ttk.Checkbutton(
            top,
            text="Use CUDA acceleration (NVIDIA)",
            variable=self.use_cuda_var,
            command=self._cuda_toggled,
        )
        self.cuda_check.grid(row=5, column=1, padx=8, pady=(7, 3), sticky="w")

        ttk.Label(top, textvariable=self.cuda_status_var).grid(
            row=5, column=2, sticky="w", padx=(4, 0)
        )

        self.run_button = ttk.Button(top, text="Run registration", command=self.run_clicked)
        self.run_button.grid(row=6, column=1, pady=(10, 2), sticky="w")

        previews = ttk.Frame(self, padding=10)
        previews.pack(fill="both", expand=True)

        self.fixed_canvas = tk.Label(
            previews, text="Fixed preview", relief="groove", width=40, anchor="center"
        )
        self.moving_canvas = tk.Label(
            previews,
            text="Select a moving image in the list to preview it",
            relief="groove",
            width=40,
            anchor="center",
        )
        self.result_canvas = tk.Label(
            previews, text="Latest result preview", relief="groove", width=40, anchor="center"
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

    def _set_running_controls(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        for widget in (
            self.fixed_button,
            self.add_moving_button,
            self.remove_moving_button,
            self.clear_moving_button,
            self.run_button,
        ):
            widget.config(state=state)

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
            f"HistRegGUI version: {__version__}\n"
            f"Build variant: {BUILD_INFO.get('variant', 'unknown')}\n"
            f"Platform: {BUILD_INFO.get('platform', 'unknown')}\n"
            f"Architecture: {BUILD_INFO.get('architecture', 'unknown')}\n"
            f"PyTorch package: {BUILD_INFO.get('torch_variant', 'unknown')}\n"
            f"PyTorch version: {torch.__version__}\n"
            f"CUDA status: {format_cuda_summary(self.cuda_info)}\n"
            f"Batch registration: one fixed target with multiple moving images\n"
            f"Supported image extensions: {supported_formats_text()}"
        )
        messagebox.showinfo("Build information", text)

    # --------------------------------------------------------------- Inputs --
    @staticmethod
    def _path_key(path: Path) -> str:
        try:
            return os.path.normcase(str(path.expanduser().resolve(strict=False)))
        except Exception:
            return os.path.normcase(str(path.expanduser().absolute()))

    def select_fixed(self) -> None:
        path = filedialog.askopenfilename(
            title="Select fixed (target) image",
            filetypes=tkinter_image_filetypes(),
        )
        if not path:
            return
        self.fixed_path = Path(path)
        self.fixed_lbl.config(text=str(self.fixed_path))
        self._update_preview(self.fixed_path, which="fixed")
        self._update_loader_status()

    def add_moving_images(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Select one or more moving images",
            filetypes=tkinter_image_filetypes(),
        )
        if not selected:
            return

        previous = {self._path_key(path) for path in self.moving_paths}
        self.moving_paths = unique_paths([*self.moving_paths, *selected])
        added = [path for path in self.moving_paths if self._path_key(path) not in previous]
        for path in added:
            self.moving_statuses[self._path_key(path)] = "Pending"

        preferred = added[0] if added else (self.moving_paths[0] if self.moving_paths else None)
        self._refresh_moving_tree(select_path=preferred)
        self._update_loader_status()
        if preferred is not None:
            self._select_moving_path(preferred, load_preview=True)

    def remove_selected_moving(self) -> None:
        selected_iids = self.moving_tree.selection()
        if not selected_iids:
            return
        remove_keys = {
            self._path_key(self.moving_tree_paths[iid])
            for iid in selected_iids
            if iid in self.moving_tree_paths
        }
        self.moving_paths = [
            path for path in self.moving_paths if self._path_key(path) not in remove_keys
        ]
        for key in remove_keys:
            self.moving_statuses.pop(key, None)
            self.moving_readers.pop(key, None)

        if self.current_moving_path and self._path_key(self.current_moving_path) in remove_keys:
            self.current_moving_path = None
            self.moving_preview_imgtk = None
            self.moving_canvas.config(
                image="", text="Select a moving image in the list to preview it"
            )

        preferred = self.moving_paths[0] if self.moving_paths else None
        self._refresh_moving_tree(select_path=preferred)
        self._update_loader_status()
        if preferred is not None:
            self._select_moving_path(preferred, load_preview=True)

    def clear_moving_images(self) -> None:
        self.moving_paths.clear()
        self.current_moving_path = None
        self.moving_statuses.clear()
        self.moving_readers.clear()
        self.moving_preview_imgtk = None
        self.moving_canvas.config(
            image="", text="Select a moving image in the list to preview it"
        )
        self.result_preview_imgtk = None
        self.result_canvas.config(image="", text="Latest result preview")
        self._refresh_moving_tree()
        self._update_loader_status()

    def _refresh_moving_tree(self, select_path: Path | None = None) -> None:
        for iid in self.moving_tree.get_children():
            self.moving_tree.delete(iid)
        self.moving_tree_paths.clear()
        self.moving_tree_iids.clear()

        for index, path in enumerate(self.moving_paths, start=1):
            key = self._path_key(path)
            reader = self.moving_readers.get(key, self._reader_for_moving(path))
            status = self.moving_statuses.get(key, "Pending")
            iid = f"moving_{index}"
            self.moving_tree.insert(
                "",
                "end",
                iid=iid,
                text=str(path),
                values=(reader, status),
            )
            self.moving_tree_paths[iid] = path
            self.moving_tree_iids[key] = iid

        count = len(self.moving_paths)
        self.moving_count_var.set(
            "No moving images selected."
            if count == 0
            else f"{count} moving image{'s' if count != 1 else ''} selected."
        )
        self.run_button.config(
            text="Run registration" if count <= 1 else f"Run registration batch ({count})"
        )

        if select_path is not None:
            iid = self.moving_tree_iids.get(self._path_key(select_path))
            if iid:
                self.moving_tree.selection_set(iid)
                self.moving_tree.focus(iid)
                self.moving_tree.see(iid)

    def _moving_selection_changed(self, _event=None) -> None:
        selection = self.moving_tree.selection()
        if not selection:
            return
        path = self.moving_tree_paths.get(selection[0])
        if path is not None and path != self.current_moving_path:
            self._select_moving_path(path, load_preview=True)

    def _select_moving_path(self, path: Path, *, load_preview: bool) -> None:
        self.current_moving_path = path
        iid = self.moving_tree_iids.get(self._path_key(path))
        if iid:
            self.moving_tree.selection_set(iid)
            self.moving_tree.focus(iid)
            self.moving_tree.see(iid)
        if load_preview:
            self._update_preview(path, which="moving")

    def _update_preview(self, path: Path, which: str) -> None:
        try:
            image, preview_info = load_preview(path, max_side=420)
            image_tk = ImageTk.PhotoImage(image)
            label_text = f"{path}  [{preview_info.summary()}]"
            if which == "fixed":
                self.fixed_preview_imgtk = image_tk
                self.fixed_canvas.config(image=image_tk, text="")
                self.fixed_lbl.config(text=label_text)
            else:
                self.moving_preview_imgtk = image_tk
                self.moving_canvas.config(image=image_tk, text="")
            self._update_loader_status()
        except Exception as exc:
            # Preview decoding is independent from registration. Keep the file
            # selected so the user can still try a manual loader.
            self._update_loader_status()
            messagebox.showwarning("Preview unavailable", f"{path}\n\n{exc}")

    def _reader_for_moving(self, moving: Path) -> str:
        if not self.fixed_path:
            key = LOADER_CHOICES.get(self.loader_var.get(), self.loader_var.get())
            return "automatic" if key == "auto" else key
        try:
            return resolve_loader_choice(self.loader_var.get(), moving, self.fixed_path)
        except Exception:
            return "error"

    def _update_loader_status(self) -> None:
        choice = self.loader_var.get()
        key = LOADER_CHOICES.get(choice, choice)
        if self.fixed_path and self.moving_paths:
            readers: list[str] = []
            errors: list[str] = []
            for moving in self.moving_paths:
                try:
                    reader = resolve_loader_choice(choice, moving, self.fixed_path)
                    readers.append(reader)
                    self.moving_readers[self._path_key(moving)] = reader
                except Exception as exc:
                    errors.append(str(exc))
                    self.moving_readers[self._path_key(moving)] = "error"

            if errors:
                self.loader_status_var.set(f"Input reader error: {errors[0]}")
            elif key == "auto":
                counts = Counter(readers)
                summary = ", ".join(
                    f"{reader} × {count}" for reader, count in sorted(counts.items())
                )
                self.loader_status_var.set(f"Input reader: auto → {summary}")
            else:
                self.loader_status_var.set(
                    f"Input reader: {key} for {len(self.moving_paths)} image(s)"
                )
            self._refresh_moving_tree(select_path=self.current_moving_path)
        else:
            self.loader_status_var.set(
                "Input reader: automatic" if key == "auto" else f"Input reader: {key}"
            )
            self._refresh_moving_tree(select_path=self.current_moving_path)

    def _set_moving_status_ui(
        self, path: Path, status: str, reader: str | None = None
    ) -> None:
        key = self._path_key(path)
        self.moving_statuses[key] = status
        if reader is not None:
            self.moving_readers[key] = reader
        iid = self.moving_tree_iids.get(key)
        if iid and self.moving_tree.exists(iid):
            current = self.moving_tree.item(iid, "values")
            current_reader = reader or (current[0] if current else self._reader_for_moving(path))
            self.moving_tree.item(iid, values=(current_reader, status))

    def _set_moving_status(
        self, path: Path, status: str, reader: str | None = None
    ) -> None:
        self._ui(self._set_moving_status_ui, path, status, reader)

    def _set_result_preview(self, image: Image.Image) -> None:
        image_tk = ImageTk.PhotoImage(image)
        self.result_preview_imgtk = image_tk
        self.result_canvas.config(image=image_tk, text="")

    # --------------------------------------------------------- Registration --
    def run_clicked(self) -> None:
        if not self.fixed_path or not self.moving_paths:
            messagebox.showwarning(
                "Missing input",
                "Please select one fixed image and at least one moving image.",
            )
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
        moving_paths = tuple(self.moving_paths)
        preset_key = self.preset_var.get()
        save_intermediate = bool(self.save_intermediate_var.get())
        loader_choice = self.loader_var.get()

        try:
            for moving in moving_paths:
                resolve_loader_choice(loader_choice, moving, fixed)
        except Exception as exc:
            messagebox.showerror("Input reader error", str(exc))
            return

        self._set_running_controls(True)
        for moving in moving_paths:
            self._set_moving_status_ui(
                moving,
                "Queued",
                resolve_loader_choice(loader_choice, moving, fixed),
            )

        threading.Thread(
            target=self._run_registration_batch,
            args=(
                fixed,
                moving_paths,
                preset_key,
                save_intermediate,
                use_cuda,
                loader_choice,
            ),
            daemon=True,
        ).start()

    def _append_error_log(
        self,
        plan: RegistrationBatchPlan,
        item: RegistrationPlanItem,
        use_cuda: bool,
        loader_key: str,
        trace: str,
    ) -> None:
        try:
            plan.error_log.parent.mkdir(parents=True, exist_ok=True)
            with plan.error_log.open("a", encoding="utf-8") as handle:
                handle.write("\n" + "=" * 80 + "\n")
                handle.write(f"[ERROR] {datetime.now().isoformat()}\n")
                handle.write(f"Fixed:  {plan.fixed_path}\n")
                handle.write(f"Moving: {item.moving_path}\n")
                handle.write(f"Output: {item.warped_output}\n")
                handle.write(f"CUDA requested: {use_cuda}\n")
                handle.write(f"Input reader: {loader_key}\n")
                handle.write(trace)
                handle.write("\n")
        except Exception:
            pass

    def _run_registration_batch(
        self,
        fixed: Path,
        moving_paths: tuple[Path, ...],
        preset_key: str,
        save_intermediate: bool,
        use_cuda: bool,
        loader_choice: str,
    ) -> None:
        plan: RegistrationBatchPlan | None = None
        try:
            run_stamp = timestamp()
            plan = build_registration_batch_plan(fixed, moving_paths, run_stamp)
            if plan.batch_root is not None:
                (plan.batch_root / "warped").mkdir(parents=True, exist_ok=True)
                (plan.batch_root / "intermediate").mkdir(parents=True, exist_ok=True)

            preset_function = PRESETS[preset_key]
            function_name = getattr(preset_function, "__name__", preset_key)
            device = "cuda:0" if use_cuda else "cpu"
            total = len(plan.items)
            results: list[dict[str, object]] = []
            successful_outputs: list[Path] = []
            failures: list[tuple[Path, str]] = []

            for item in plan.items:
                started_at = datetime.now().isoformat()
                loader_key = resolve_loader_choice(
                    loader_choice, item.moving_path, fixed
                )
                result: dict[str, object] = {
                    "index": item.index,
                    "status": "failed",
                    "fixed_image": str(fixed),
                    "moving_image": str(item.moving_path),
                    "warped_output": str(item.warped_output),
                    "intermediate_directory": str(item.run_directory),
                    "loader": loader_key,
                    "device": device,
                    "preset": function_name,
                    "started_at": started_at,
                    "finished_at": "",
                    "error": "",
                }

                self._set_moving_status(
                    item.moving_path,
                    f"Running {item.index}/{total}",
                    loader_key,
                )
                if use_cuda:
                    gpu_name = self.cuda_info.device_names[0]
                    self._set_status(
                        f"[{item.index}/{total}] Registering {item.moving_path.name} "
                        f"with {loader_key} on CUDA: {gpu_name} ..."
                    )
                else:
                    self._set_status(
                        f"[{item.index}/{total}] Registering {item.moving_path.name} "
                        f"with {loader_key} on CPU ..."
                    )

                try:
                    item.warped_output.parent.mkdir(parents=True, exist_ok=True)
                    item.run_directory.mkdir(parents=True, exist_ok=True)

                    parameters = configure_registration_device(
                        preset_function(), device  # type: ignore[operator]
                    )
                    parameters = configure_registration_loader(parameters, loader_key)
                    loader = pick_loader_for_warp(loader_key)

                    registration = (
                        deeperhistreg.direct_registration.DeeperHistReg_FullResolution(
                            registration_parameters=parameters
                        )
                    )
                    registration.run_registration(
                        str(item.moving_path), str(fixed), str(item.run_directory)
                    )

                    self._set_status(
                        f"[{item.index}/{total}] Finding displacement field for "
                        f"{item.moving_path.name} ..."
                    )
                    displacement_candidates = (
                        list(item.run_directory.rglob("displacement_field.mha"))
                        + list(item.run_directory.rglob("*disp*.mha"))
                        + list(item.run_directory.rglob("*.mha"))
                    )
                    if not displacement_candidates:
                        raise FileNotFoundError(
                            "No displacement field was found under: "
                            f"{item.run_directory}"
                        )
                    displacement_field = next(
                        (
                            path
                            for path in displacement_candidates
                            if path.name.lower() == "displacement_field.mha"
                        ),
                        displacement_candidates[0],
                    )

                    self._set_status(
                        f"[{item.index}/{total}] Warping and saving "
                        f"{item.warped_output.name} ..."
                    )
                    deeperhistreg.apply_deformation(
                        source_image_path=str(item.moving_path),
                        target_image_path=str(fixed),
                        warped_image_path=str(item.warped_output),
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
                        result_image, _result_info = load_preview(
                            item.warped_output, max_side=420
                        )
                        self._ui(self._set_result_preview, result_image)
                    except Exception:
                        pass

                    if not save_intermediate:
                        shutil.rmtree(item.run_directory, ignore_errors=True)

                    result["status"] = "success"
                    successful_outputs.append(item.warped_output)
                    self._set_moving_status(
                        item.moving_path, "Done", loader_key
                    )

                except Exception as exc:
                    trace = traceback.format_exc()
                    result["error"] = f"{exc!r}"
                    failures.append((item.moving_path, str(exc)))
                    self._append_error_log(
                        plan, item, use_cuda, loader_key, trace
                    )
                    short_error = str(exc).strip() or type(exc).__name__
                    if len(short_error) > 90:
                        short_error = short_error[:87] + "..."
                    self._set_moving_status(
                        item.moving_path, f"Failed: {short_error}", loader_key
                    )
                finally:
                    result["finished_at"] = datetime.now().isoformat()
                    results.append(result)
                    # Sequential processing is deliberate. Releasing tensors and
                    # cached GPU blocks between images avoids batch-wide memory
                    # growth, especially for large whole-slide registrations.
                    gc.collect()
                    if use_cuda:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

            if plan.batch_root is not None and not save_intermediate:
                intermediate_root = plan.batch_root / "intermediate"
                try:
                    if intermediate_root.exists() and not any(intermediate_root.iterdir()):
                        intermediate_root.rmdir()
                except Exception:
                    pass

            write_registration_manifest(plan, results)

            success_count = len(successful_outputs)
            failure_count = len(failures)
            if plan.is_batch:
                destination_text = f"Batch folder:\n{plan.batch_root}"
            elif successful_outputs:
                destination_text = f"Warped image:\n{successful_outputs[0]}"
            else:
                destination_text = f"Output folder:\n{fixed.parent}"

            summary = (
                f"Completed {len(plan.items)} registration(s).\n\n"
                f"Successful: {success_count}\n"
                f"Failed: {failure_count}\n\n"
                f"{destination_text}\n\n"
                f"CSV manifest:\n{plan.manifest_csv}\n"
                f"JSON manifest:\n{plan.manifest_json}\n\n"
                f"Execution device: {device}"
            )
            if failures:
                summary += f"\n\nError log:\n{plan.error_log}"

            if failure_count == 0:
                self._set_status(
                    f"Done. {success_count} registration(s) saved. Manifest: "
                    f"{plan.manifest_csv}"
                )
                self._ui(messagebox.showinfo, "Registration complete", summary)
            elif success_count > 0:
                self._set_status(
                    f"Completed with errors: {success_count} succeeded, "
                    f"{failure_count} failed."
                )
                self._ui(
                    messagebox.showwarning,
                    "Batch completed with errors",
                    summary,
                )
            else:
                self._set_status("All registrations failed.")
                self._ui(messagebox.showerror, "Registration failed", summary)

        except Exception as exc:
            trace = traceback.format_exc()
            self._set_status("Error.")
            log_path = plan.error_log if plan is not None else fixed.parent / "HistRegGUI_error.log"
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write("\n" + "=" * 80 + "\n")
                    handle.write(f"[BATCH ERROR] {datetime.now().isoformat()}\n")
                    handle.write(trace)
                    handle.write("\n")
            except Exception:
                pass
            self._show_error(
                "Registration error",
                "The registration batch could not be completed.\n\n"
                f"Error: {exc!r}\n\n"
                f"Full traceback:\n{trace}\n\n"
                f"Log saved to:\n{log_path}",
            )
        finally:
            self._ui(self._set_running_controls, False)

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

    # Verify both the canonical private name and the compatibility alias used
    # by older packaging diagnostics.
    finder_module = install_pillow_tkinter_finder_alias()
    if finder_module is None or "PIL.tkinter_finder" not in sys.modules:
        raise RuntimeError("Pillow Tk finder compatibility alias was not installed.")

    info = detect_cuda(torch, probe=False)
    batch_probe = build_registration_batch_plan(
        Path("fixed.tif"), [Path("moving_a.tif"), Path("moving_b.svs")], "self_test"
    )
    if len(batch_probe.items) != 2 or not batch_probe.is_batch:
        raise RuntimeError("Batch registration planning is unavailable.")

    payload = {
        "status": "ok",
        "version": __version__,
        "preset_count": len(PRESETS),
        "build": BUILD_INFO,
        "torch_version": str(torch.__version__),
        "cuda_compiled": info.compiled_with_cuda,
        "cuda_available": info.available,
        "supported_extension_count": len(supported_formats_text().split(", ")),
        "pillow_tkinter_finder": "ok",
        "pillow_tkinter_finder_alias": "ok",
        "batch_registration": "ok",
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
