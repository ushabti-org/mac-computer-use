from setuptools import setup, find_packages

setup(
    name="computer-use-demo",
    version="0.0.1",
    packages=find_packages(include=['computer_use_demo', 'computer_use_demo.*']),
    install_requires=[
        'anthropic',
        'pillow',
        'pyautogui',
    ],
) 