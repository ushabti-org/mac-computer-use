# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Add the computer-use-demo directory to the path
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

a = Analysis(
    ['mac-computer-use.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('icons', 'icons'),
        ('plugins', 'plugins'),
        ('computer-use-demo/computer_use_demo', 'computer_use_demo'),
    ],
    hiddenimports=[
        'PIL',
        'computer_use_demo',
        'computer_use_demo.loop',
        'computer_use_demo.tools',
        'computer_use_demo.tools.base',
        'computer_use_demo.tools.bash',
        'computer_use_demo.tools.collection',
        'computer_use_demo.tools.computer',
        'computer_use_demo.tools.edit',
        'computer_use_demo.tools.run',
        'util',
        'PySide6',
        'openai',
        'tiktoken',
        'gtts',
        'pyqtdarktheme',
        'pyautogui',
        'anthropic',
        'platformdirs',
        'darkdetect',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='mac-computer-use',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity="Developer ID Application: Thomas Joseph Shelley (K9T7D9TCVL)",
    entitlements_file="entitlements.plist",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='mac-computer-use'
)

app = BUNDLE(
    coll,
    name='mac-computer-use.app',
    icon='./icons/mac-computer-use.icns',
    bundle_identifier='com.thomasshelley.mac-computer-use',
    info_plist={
        'CFBundleShortVersionString': '0.0.0',
        'CFBundleVersion': '0.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.13.0',
    }
) 