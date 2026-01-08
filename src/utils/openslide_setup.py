import os

OPENSLIDE_BIN = r"C:\openslide-bin-4.0.0.11-windows-x64\bin"

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(OPENSLIDE_BIN)
