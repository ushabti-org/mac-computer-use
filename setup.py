"""
This is a setup.py script generated by py2applet

Usage:
    python setup.py py2app
"""

from setuptools import setup

APP = ['mac-computer-use.py']
DATA_FILES = []
OPTIONS = {
    'iconfile': './icons/mac-computer-use.icns',
    # requests wants (but doesn't _need_?) 'charset_normalizer' or 'chardet' - it gives a warning at runtime, including them here doesn't seem to fix it though...
    'includes': ['charset_normalizer', 'chardet'],
    # rubicon is a dep of mouseinfo->pyautogui I believe, TODO excluding it may break that... (but throws a build time error if you put it in includes or specify nothing)
    'excludes': ['rubicon'],
    'packages': ['PIL', 'computer-use-demo', 'util'],

}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
