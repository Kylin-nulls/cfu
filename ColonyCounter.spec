# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


block_cipher = None
project_dir = Path.cwd()

datas = [
    (str(project_dir / "app.py"), "."),
    (str(project_dir / "colony_counter.py"), "."),
    (str(project_dir / "README.md"), "."),
]

sample_dir = project_dir / "samples"
if sample_dir.exists():
    datas.append((str(sample_dir), "samples"))

binaries = []
hiddenimports = [
    "colony_counter",
    "streamlit.web.bootstrap",
    "streamlit.runtime.scriptrunner.script_runner",
]

for package in [
    "streamlit",
    "altair",
    "pydeck",
    "tornado",
    "pandas",
    "numpy",
    "PIL",
    "cv2",
    "pyarrow",
    "jsonschema",
    "protobuf",
    "watchdog",
]:
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package)
    except Exception:
        continue
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports


a = Analysis(
    ["desktop_launcher.py"],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ColonyCounter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ColonyCounter",
)
