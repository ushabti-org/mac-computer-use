import config
from configDefault import *
import os

import platform, ctypes, glob, traceback
import pyautogui
from functools import partial
import sys, pprint, qdarktheme
from shutil import copyfile
# try:
#     from pocketsphinx import LiveSpeech, get_model_path
#     isPocketsphinxInstalled = True
# except:
#     isPocketsphinxInstalled = False

from datetime import datetime
if config.qtLibrary == "pyside6":
    from PySide6.QtPrintSupport import QPrinter, QPrintDialog
    from PySide6.QtCore import Qt, QThread, Signal, QRegularExpression, QEvent, QObject, Slot
    from PySide6.QtGui import QStandardItemModel, QStandardItem, QGuiApplication, QAction, QIcon, QFontMetrics, QTextDocument, QClipboard, QImage
    from PySide6.QtWidgets import QCompleter, QMenu, QSystemTrayIcon, QApplication, QMainWindow, QWidget, QDialog, QFileDialog, QDialogButtonBox, QFormLayout, QLabel, QMessageBox, QCheckBox, QPlainTextEdit, QProgressBar, QPushButton, QListView, QHBoxLayout, QVBoxLayout, QLineEdit, QSplitter, QComboBox
else:
    from qtpy.QtPrintSupport import QPrinter, QPrintDialog
    from qtpy.QtCore import Qt, QThread, Signal, QRegularExpression, QEvent, QObject, Slot
    from qtpy.QtGui import QStandardItemModel, QStandardItem, QGuiApplication, QIcon, QFontMetrics, QTextDocument, QClipboard
    from qtpy.QtWidgets import QCompleter, QMenu, QSystemTrayIcon, QApplication, QMainWindow, QAction, QWidget, QDialog, QFileDialog, QDialogButtonBox, QFormLayout, QLabel, QMessageBox, QCheckBox, QPlainTextEdit, QProgressBar, QPushButton, QListView, QHBoxLayout, QVBoxLayout, QLineEdit, QSplitter, QComboBox


# for QtAsync
# import PySide6.QtAsyncio as QtAsyncio
import asyncio
import random
import outcome, traceback, signal, trio
# from contextlib import contextmanager
# import importlib  
# loop = importlib.import_module("computer-use-demo.computer_use_demo.loop")
# sampling_loop = loop.sampling_loop  # Access the function directly from the imported module
# tools = importlib.import_module("computer-use-demo.computer_use_demo.tools")
# from tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

from enum import StrEnum

from chat_widget import ChatGPTAPI 

from platformdirs import user_data_dir
wd = user_data_dir(config.getAppName())

try:
    width, height = pyautogui.size()
    os.environ["HEIGHT"] = str(height)
    os.environ["WIDTH"] = str(width)
    os.environ["DISPLAY_NUM"] = "1"
    print("DIMENSIONS")
    print(str(height))
    print(str(width))
except ImportError:
    print("PyAutoGUI not installed. Using default screen dimensions.")
    os.environ["HEIGHT"] = "768"
    os.environ["WIDTH"] = "1024"
    os.environ["DISPLAY_NUM"] = "1"

class AsyncHelper(QObject):

    class ReenterQtObject(QObject):
        """ This is a QObject to which an event will be posted, allowing
            Trio to resume when the event is handled. event.fn() is the
            next entry point of the Trio event loop. """
        def event(self, event):
            print("ReenterQtObject event")
            if event.type() == QEvent.Type.User + 1:
                event.fn()
                return True
            return False

    class ReenterQtEvent(QEvent):
        """ This is the QEvent that will be handled by the ReenterQtObject.
            self.fn is the next entry point of the Trio event loop. """
        def __init__(self, fn):
            print("ReenterQtEvent init")
            super().__init__(QEvent.Type(QEvent.Type.User + 1))
            self.fn = fn

    def __init__(self, worker, entry):
        print("AsyncHelper init " + str(worker) + " " + str(entry))
        super().__init__()
        self.reenter_qt = self.ReenterQtObject()
        self.entry = entry

        self.worker = worker
        if hasattr(self.worker, "start_signal") and isinstance(self.worker.start_signal, Signal):
            print("register start_signal " + str(self.worker.start_signal))
            self.worker.start_signal.connect(self.launch_guest_run)
        else:
            print("failed to register start_signal")

    @Slot()
    def launch_guest_run(self):
        """ To use Trio and Qt together, one must run the Trio event
            loop as a "guest" inside the Qt "host" event loop. """
        print("launch_guest_run")
        if not self.entry:
            print("this is not good")
            raise Exception("No entry point for the Trio guest run was set.")
        trio.lowlevel.start_guest_run(
            self.entry,
            run_sync_soon_threadsafe=self.next_guest_run_schedule,
            done_callback=self.trio_done_callback,
        )

    def next_guest_run_schedule(self, fn):
        """ This function serves to re-schedule the guest (Trio) event
            loop inside the host (Qt) event loop. It is called by Trio
            at the end of an event loop run in order to relinquish back
            to Qt's event loop. By posting an event on the Qt event loop
            that contains Trio's next entry point, it ensures that Trio's
            event loop will be scheduled again by Qt. """
        print("next_guest_run_schedule")
        QApplication.postEvent(self.reenter_qt, self.ReenterQtEvent(fn))

    def trio_done_callback(self, outcome_):
        """ This function is called by Trio when its event loop has
            finished. """
        print("trio_done_callback")
        if isinstance(outcome_, outcome.Error):
            error = outcome_.error
            traceback.print_exception(type(error), error, error.__traceback__)

class MainWindow(QMainWindow):
    start_signal = Signal()

    def __init__(self):
        super().__init__()
        self.initUI()

    async def some_async_function(self):
        try:
            print("some_async_function pre")
            # await asyncio.sleep(3)
            await trio.sleep(3)
            self.setWindowTitle("window reset + " + str(random.random()))
            # self.chatGPT.setText("What do you get if you multiply six by nine?"))

            print("some_async_function post")
            # await asyncio.sleep(3)
            await trio.sleep(3)
            self.setWindowTitle("window reset + " + str(random.random()))

            print("some_async_function post post")
        except Exception as e:
            print(f"Error in some_async_function: {str(e)}")
            traceback.print_exc()

    async def run_the_loop(self):
        print("run_the_loop")
        return await self.chatGPT.process_computer_use(self.chatGPT.last_user_input)
        # return await self.some_async_function()

    @Slot()
    def async_start(self):
        print("async_start " + str(self.start_signal))
        self.start_signal.emit()

    def reloadMenubar(self):
        self.menuBar().clear()
        self.createMenubar()

    def createMenubar(self):
        # Create a menu bar
        menubar = self.menuBar()

        # Create a File menu and add it to the menu bar
        file_menu = menubar.addMenu(config.thisTranslation["chat"])

        # Add test button
        test_action = QAction("Test PyAutoGUI", self)
        test_action.triggered.connect(self.testPyAutoGUI)
        file_menu.addAction(test_action)

        screenshot_action = QAction("Take Screenshot", self)
        screenshot_action.setShortcut("Ctrl+Shift+S")  # Optional: Add keyboard shortcut
        screenshot_action.triggered.connect(self.takeScreenshot)
        file_menu.addAction(screenshot_action)

        new_action = QAction(config.thisTranslation["openDatabase"], self)
        new_action.setShortcut("Ctrl+Shift+O")
        new_action.triggered.connect(self.chatGPT.openDatabase)
        file_menu.addAction(new_action)

        new_action = QAction(config.thisTranslation["newDatabase"], self)
        new_action.setShortcut("Ctrl+Shift+N")
        new_action.triggered.connect(self.chatGPT.newDatabase)
        file_menu.addAction(new_action)

        new_action = QAction(config.thisTranslation["saveDatabaseAs"], self)
        new_action.setShortcut("Ctrl+Shift+S")
        new_action.triggered.connect(lambda: self.chatGPT.newDatabase(copyExistingDatabase=True))
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        new_action = QAction(config.thisTranslation["fileManager"], self)
        new_action.triggered.connect(self.openDatabaseDirectory)
        file_menu.addAction(new_action)

        new_action = QAction(config.thisTranslation["pluginDirectory"], self)
        new_action.triggered.connect(self.openPluginsDirectory)
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        new_action = QAction(config.thisTranslation["newChat"], self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.chatGPT.newData)
        file_menu.addAction(new_action)

        new_action = QAction(config.thisTranslation["saveChat"], self)
        new_action.setShortcut("Ctrl+S")
        new_action.triggered.connect(self.chatGPT.saveData)
        file_menu.addAction(new_action)

        new_action = QAction(config.thisTranslation["exportChat"], self)
        new_action.triggered.connect(self.chatGPT.exportData)
        file_menu.addAction(new_action)

        new_action = QAction(config.thisTranslation["printChat"], self)
        new_action.setShortcut("Ctrl+P")
        new_action.triggered.connect(self.chatGPT.printData)
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        new_action = QAction(config.thisTranslation["readTextFile"], self)
        new_action.triggered.connect(self.chatGPT.openTextFileDialog)
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        new_action = QAction(config.thisTranslation["countPromptTokens"], self)
        new_action.triggered.connect(self.chatGPT.num_tokens_from_messages)
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        # Create a Exit action and add it to the File menu
        exit_action = QAction(config.thisTranslation["exit"], self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip(config.thisTranslation["exitTheApplication"])
        exit_action.triggered.connect(QGuiApplication.instance().quit)
        file_menu.addAction(exit_action)

        # Create customise menu
        customise_menu = menubar.addMenu(config.thisTranslation["customise"])

        openSettings = QAction(config.thisTranslation["configure"], self)
        openSettings.triggered.connect(self.chatGPT.showSettingsDialog)
        customise_menu.addAction(openSettings)

        customise_menu.addSeparator()

        new_action = QAction(config.thisTranslation["toggleDarkTheme"], self)
        new_action.triggered.connect(self.toggleTheme)
        customise_menu.addAction(new_action)

        new_action = QAction(config.thisTranslation["toggleSystemTray"], self)
        new_action.triggered.connect(self.toggleSystemTray)
        customise_menu.addAction(new_action)

        # new_action = QAction(config.thisTranslation["toggleMultilineInput"], self)
        # new_action.setShortcut("Ctrl+L")
        # new_action.triggered.connect(self.chatGPT.multilineButtonClicked)
        # customise_menu.addAction(new_action)

        new_action = QAction(config.thisTranslation["toggleRegexp"], self)
        new_action.setShortcut("Ctrl+E")
        new_action.triggered.connect(self.toggleRegexp)
        customise_menu.addAction(new_action)

        new_action = QAction(config.thisTranslation["toggleComputerUse"], self)
        new_action.triggered.connect(self.toggleComputerUse)
        customise_menu.addAction(new_action)

        # Create predefined context menu
        context_menu = menubar.addMenu(config.thisTranslation["predefinedContext"])
        for index, context in enumerate(config.predefinedContexts):
            contextAction = QAction(context, self)
            if index < 10:
                contextAction.setShortcut(f"Ctrl+{index}")
            contextAction.triggered.connect(partial(self.chatGPT.bibleChatAction, context))
            context_menu.addAction(contextAction)

        # Create a plugin menu
        plugin_menu = menubar.addMenu(config.thisTranslation["plugins"])

        pluginFolder = os.path.join(os.getcwd(), "plugins")
        for index, plugin in enumerate(self.fileNamesWithoutExtension(pluginFolder, "py")):
            new_action = QAction(plugin, self)
            new_action.setCheckable(True)
            new_action.setChecked(False if plugin in config.chatGPTPluginExcludeList else True)
            new_action.triggered.connect(partial(self.updateExcludePluginList, plugin))
            plugin_menu.addAction(new_action)

        # Create a text selection menu
        text_selection_menu = menubar.addMenu(config.thisTranslation["textSelection"])

        # new_action = QAction(config.thisTranslation["webBrowser"], self)
        # new_action.triggered.connect(self.chatGPT.webBrowse)
        # text_selection_menu.addAction(new_action)

        new_action = QAction(config.thisTranslation["runAsPythonCommand"], self)
        new_action.triggered.connect(self.chatGPT.runPythonCommand)
        text_selection_menu.addAction(new_action)

        new_action = QAction(config.thisTranslation["runAsSystemCommand"], self)
        new_action.triggered.connect(self.chatGPT.runSystemCommand)
        text_selection_menu.addAction(new_action)


    def initUI(self):
        # Set a central widget
        self.chatGPT = ChatGPTAPI(self)
        self.setCentralWidget(self.chatGPT)

        # TODO REMOVE - TESTING FOR ASYNC
        # from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget)
        # widget = QWidget()
        # self.setCentralWidget(widget)
        # layout = QVBoxLayout(widget)
        # self.text = QLabel("The answer is 42.")
        # layout.addWidget(self.text, alignment=Qt.AlignmentFlag.AlignCenter)
        # print("registering button")
        # async_trigger = QPushButton(text="What is the question?")
        # async_trigger.clicked.connect(self.async_start)
        # layout.addWidget(async_trigger, alignment=Qt.AlignmentFlag.AlignCenter)
        # print("post-registering button")
        # END TESTING FOR ASYNC

        # create menu bar
        self.createMenubar()

        # set initial window size
        self.setWindowTitle("mac-computer-use")
        # self.resize(QGuiApplication.primaryScreen().availableSize() * 1 / 4)
        defaultWidth = 360  # default width in pixels
        defaultHeight = 720  # default height in pixels
        self.resize(defaultWidth, defaultHeight)
        self.show()

    def updateExcludePluginList(self, plugin):
        if plugin in config.chatGPTPluginExcludeList:
            config.chatGPTPluginExcludeList.remove(plugin)
        else:
            config.chatGPTPluginExcludeList.append(plugin)
        internetSeraches = "integrate google searches"
        if internetSeraches in config.chatGPTPluginExcludeList and config.loadingInternetSearches == "auto":
            config.loadingInternetSearches = "none"
        elif not internetSeraches in config.chatGPTPluginExcludeList and config.loadingInternetSearches == "none":
            config.loadingInternetSearches = "auto"
            config.chatGPTApiFunctionCall = "auto"
        # reload plugins
        config.chatGPTApi.runPlugins()

    def fileNamesWithoutExtension(self, dir, ext):
        files = glob.glob(os.path.join(dir, "*.{0}".format(ext)))
        return sorted([file[len(dir)+1:-(len(ext)+1)] for file in files if os.path.isfile(file)])

    def getOpenCommand(self):
        thisOS = platform.system()
        if thisOS == "Windows":
            openCommand = "start"
        elif thisOS == "Darwin":
            openCommand = "open"
        elif thisOS == "Linux":
            openCommand = "xdg-open"
        return openCommand

    def openDatabaseDirectory(self):
        databaseDirectory = os.path.dirname(os.path.abspath(config.chatGPTApiLastChatDatabase))
        openCommand = self.getOpenCommand()
        os.system(f"{openCommand} {databaseDirectory}")

    def openPluginsDirectory(self):
        openCommand = self.getOpenCommand()
        os.system(f"{openCommand} plugins")

    def toggleRegexp(self):
        config.regexpSearchEnabled = not config.regexpSearchEnabled
        self.chatGPT.updateSearchToolTips()
        QMessageBox.information(self, "mac-computer-use", f"Regular expression for search and replace is {'enabled' if config.regexpSearchEnabled else 'disabled'}!")

    def toggleSystemTray(self):
        config.enableSystemTray = not config.enableSystemTray
        QMessageBox.information(self, "mac-computer-use", "You need to restart this application to make the changes effective.")

    def toggleTheme(self):
        config.darkTheme = not config.darkTheme
        qdarktheme.setup_theme() if config.darkTheme else qdarktheme.setup_theme("light")

    def toggleComputerUse(self):
        config.computerUseEnabled = not getattr(config, "computerUseEnabled", False)
        QMessageBox.information(self, "mac-computer-use", f"Computer use is {'enabled' if config.computerUseEnabled else 'disabled'}!")

    # Work with system tray
    def isWayland(self):
        if platform.system() == "Linux" and not os.getenv('QT_QPA_PLATFORM') is None and os.getenv('QT_QPA_PLATFORM') == "wayland":
            return True
        else:
            return False

    def bringToForeground(self, window):
        if window and not (window.isVisible() and window.isActiveWindow()):
            window.raise_()
            # Method activateWindow() does not work with qt.qpa.wayland
            # platform.system() == "Linux" and not os.getenv('QT_QPA_PLATFORM') is None and os.getenv('QT_QPA_PLATFORM') == "wayland"
            # The error message is received when QT_QPA_PLATFORM=wayland:
            # qt.qpa.wayland: Wayland does not support QWindow::requestActivate()
            # Therefore, we use hide and show methods instead with wayland.
            if window.isVisible() and not window.isActiveWindow():
                window.hide()
            window.show()
            if not self.isWayland():
                window.activateWindow()

    def testPyAutoGUI(self):
        try:
            # Move mouse to 10,10 and click
            pyautogui.click(x=10, y=10)
        except ImportError:
            QMessageBox.warning(self, "Missing Dependency", "Please install PyAutoGUI:\npip install pyautogui")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to perform mouse action: {str(e)}")

    def takeScreenshot(self):
        try:
            import pyautogui
            from PIL import Image
            import io
            from PySide6.QtGui import QClipboard, QImage  # or from qtpy.QtGui if using qtpy

            # Take screenshot
            screenshot = pyautogui.screenshot()
            
            # Convert PIL image to QImage
            with io.BytesIO() as bio:
                screenshot.save(bio, 'PNG')
                img_data = bio.getvalue()
                qimg = QImage.fromData(img_data)

            # Copy to clipboard
            clipboard = QGuiApplication.clipboard()
            clipboard.setImage(qimg)
            
            QMessageBox.information(self, "Success", "Screenshot copied to clipboard!")
        except ImportError:
            QMessageBox.warning(self, "Missing Dependency", "Please install PyAutoGUI:\npip install pyautogui")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to take screenshot: {str(e)}")


if __name__ == '__main__':
    def showMainWindow():
        if not hasattr(config, "mainWindow") or config.mainWindow is None:
            print("ERROR TRYING TO INIT SHOW MAIN WINDOW BUT WE DON'T DO THIS HERE ANYMORE")
            # print("showMainWindow init")
            # config.mainWindow = MainWindow()
            # async_helper = AsyncHelper(config.mainWindow, config.mainWindow.some_async_function)
            # config.mainWindow.show()
            # qdarktheme.setup_theme() if config.darkTheme else qdarktheme.setup_theme("light")
        else:
            print("showMainWindow bringToForeground")
            config.mainWindow.bringToForeground(config.mainWindow)

    def aboutToQuit():
        with open(config.getConfigPath(), "w", encoding="utf-8") as fileObj:
            for name in dir(config):
                excludeFromSavingList = (
                    "mainWindow", # main window object
                    "chatGPTApi", # GUI object
                    "chatGPTTransformers", # used with plugins; transform ChatGPT response message
                    "predefinedContexts", # used with plugins; pre-defined contexts
                    "inputSuggestions", # used with plugins; user input suggestions
                    "integrate_google_searches_signature",
                    "chatGPTApiFunctionSignatures", # used with plugins; function calling
                    "chatGPTApiAvailableFunctions", # used with plugins; function calling
                    "pythonFunctionResponse", # used with plugins; function calling when function name is 'python',
                    "setupConfig", # this is an internal function of config.py
                    "getConfigPath", # this is an internal function of config.py
                    "getChatsPath", # this is an internal function of config.py
                    "getAppName", # this is an internal function of config.py
                )
                if not name.startswith("__") and not name in excludeFromSavingList:
                    try:
                        value = eval(f"config.{name}")
                        fileObj.write("{0} = {1}\n".format(name, pprint.pformat(value)))
                    except:
                        pass

    thisOS = platform.system()
    appName = "mac-computer-use"
    # Windows icon
    if thisOS == "Windows":
        myappid = "mac-computer-use.gui"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        windowsIconPath = os.path.abspath(os.path.join(sys.path[0], "icons", f"{appName}.ico"))
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(windowsIconPath)
    # app
    qdarktheme.enable_hi_dpi()
    app = QApplication(sys.argv)
    iconPath = os.path.abspath(os.path.join(sys.path[0], "icons", f"{appName}.png"))
    appIcon = QIcon(iconPath)
    app.setWindowIcon(appIcon)
    # showMainWindow()
    config.mainWindow = MainWindow()
    # async_helper = AsyncHelper(config.mainWindow, config.mainWindow.some_async_function)
    async_helper = AsyncHelper(config.mainWindow, config.mainWindow.run_the_loop)
    config.mainWindow.show()
    qdarktheme.setup_theme() if config.darkTheme else qdarktheme.setup_theme("light")
    # connection
    app.aboutToQuit.connect(aboutToQuit)

    # Desktop shortcut
    # on Windows
    if thisOS == "Windows":
        desktopPath = os.path.join(os.path.expanduser('~'), 'Desktop')
        shortcutDir = desktopPath if os.path.isdir(desktopPath) else wd
        shortcutBat1 = os.path.join(shortcutDir, f"{appName}.bat")
        shortcutCommand1 = f'''powershell.exe -NoExit -Command "python '{thisFile}'"'''
        # Create .bat for application shortcuts
        if not os.path.exists(shortcutBat1):
            try:
                with open(shortcutBat1, "w") as fileObj:
                    fileObj.write(shortcutCommand1)
            except:
                pass
    # on macOS
    elif thisOS == "Darwin":
        shortcut_file = os.path.expanduser(f"~/Desktop/{appName}.command")
        if not os.path.isfile(shortcut_file):
            with open(shortcut_file, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(f"cd {wd}\n")
                f.write(f"{sys.executable} {thisFile} gui\n")
            os.chmod(shortcut_file, 0o755)
    # additional shortcuts on Linux
    elif thisOS == "Linux":
        def desktopFileContent():
            iconPath = os.path.join(wd, "icons", "mac-computer-use.png")
            return """#!/usr/bin/env xdg-open

[Desktop Entry]
Version=1.0
Type=Application
Terminal=false
Path={0}
Exec={1} {2}
Icon={3}
Name=mac-computer-use
""".format(wd, sys.executable, thisFile, iconPath)

        ubaLinuxDesktopFile = os.path.join(wd, f"{appName}.desktop")
        if not os.path.exists(ubaLinuxDesktopFile):
            # Create .desktop shortcut
            with open(ubaLinuxDesktopFile, "w") as fileObj:
                fileObj.write(desktopFileContent())
            try:
                # Try to copy the newly created .desktop file to:
                from pathlib import Path
                # ~/.local/share/applications
                userAppDir = os.path.join(str(Path.home()), ".local", "share", "applications")
                userAppDirShortcut = os.path.join(userAppDir, f"{appName}.desktop")
                if not os.path.exists(userAppDirShortcut):
                    Path(userAppDir).mkdir(parents=True, exist_ok=True)
                    copyfile(ubaLinuxDesktopFile, userAppDirShortcut)
                # ~/Desktop
                homeDir = os.environ["HOME"]
                desktopPath = f"{homeDir}/Desktop"
                desktopPathShortcut = os.path.join(desktopPath, f"{appName}.desktop")
                if os.path.isfile(desktopPath) and not os.path.isfile(desktopPathShortcut):
                    copyfile(ubaLinuxDesktopFile, desktopPathShortcut)
            except:
                pass

    # system tray
    if config.enableSystemTray:
        app.setQuitOnLastWindowClosed(False)
        # Set up tray icon
        tray = QSystemTrayIcon()
        tray.setIcon(appIcon)
        tray.setToolTip("mac-computer-use")
        tray.setVisible(True)
        # Import system tray menu
        trayMenu = QMenu()
        showMainWindowAction = QAction(config.thisTranslation["show"])
        showMainWindowAction.triggered.connect(showMainWindow)
        trayMenu.addAction(showMainWindowAction)
        # Add a separator
        trayMenu.addSeparator()
        # Quit
        quitAppAction = QAction(config.thisTranslation["exit"])
        quitAppAction.triggered.connect(app.quit)
        trayMenu.addAction(quitAppAction)
        tray.setContextMenu(trayMenu)

    # run the app
    # how does this work without passing app? it just do bro
    # QtAsyncio.run(handle_sigint=True)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec() if config.qtLibrary == "pyside6" else app.exec_())
