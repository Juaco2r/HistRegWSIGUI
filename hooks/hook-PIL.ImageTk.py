"""PyInstaller hook for Pillow's dynamically imported Tk bridge."""

hiddenimports = [
    "PIL._tkinter_finder",
    "PIL._imagingtk",
    "tkinter",
    "_tkinter",
]
