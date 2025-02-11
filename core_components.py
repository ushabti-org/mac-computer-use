import config
from configDefault import *

import os

import re, sqlite3
from platformdirs import user_data_dir
try:
    from pocketsphinx import LiveSpeech, get_model_path
    isPocketsphinxInstalled = True
except:
    isPocketsphinxInstalled = False

if config.qtLibrary == "pyside6":
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QCheckBox, QLineEdit, QComboBox
else:
    from qtpy.QtCore import Qt, QThread, Signal
    from qtpy.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QCheckBox, QLineEdit, QComboBox

# import httpx
# from contextlib import contextmanager
# import importlib  
# loop = importlib.import_module("computer-use-demo.computer_use_demo.loop")
# sampling_loop = loop.sampling_loop  # Access the function directly from the imported module
# tools = importlib.import_module("computer-use-demo.computer_use_demo.tools")
# from tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

# from enum import StrEnum

# TODO this seems wrong - need to read this from somewhere?
# anthropic tools get initialized based on these
# os.environ["HEIGHT"] = "768"
# os.environ["WIDTH"] = "1024" 
# os.environ["DISPLAY_NUM"] = "1"


# def read_secret(filename):
#     try:
#         with open(os.path.join("SECRETS", filename), 'r') as f:
#             return f.read().strip()
#     except:
#         return ""

class SpeechRecognitionThread(QThread):
    phrase_recognized = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Load API keys from SECRETS
        # openai_key = read_secret("openai-api-key.txt")
        # anthropic_key = read_secret("anthropic-api-key.txt")
        
        # self.apiKeyEdit = QLineEdit(config.openaiApiKey or openai_key)
        # self.apiKeyEdit.setEchoMode(QLineEdit.Password)
        # self.anthropicKeyEdit = QLineEdit(getattr(config, "anthropicApiKey", anthropic_key))
        # self.anthropicKeyEdit.setEchoMode(QLineEdit.Password)

    def run(self):
        self.is_running = True
        if config.pocketsphinxModelPath:
            # download English dictionary at: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
            # download voice models at https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/
            speech = LiveSpeech(
                #sampling_rate=16000,  # optional
                hmm=get_model_path(config.pocketsphinxModelPath),
                lm=get_model_path(config.pocketsphinxModelPathBin),
                dic=get_model_path(config.pocketsphinxModelPathDict),
            )
        else:
            speech = LiveSpeech()

        for phrase in speech:
            if not self.is_running:
                break
            recognized_text = str(phrase)
            self.phrase_recognized.emit(recognized_text)

    def stop(self):
        self.is_running = False


# class ApiDialog(QDialog):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle(config.thisTranslation["settings"])

#         self.apiKeyEdit = QLineEdit(config.openaiApiKey)
#         self.apiKeyEdit.setEchoMode(QLineEdit.Password)
#         self.orgEdit = QLineEdit(config.openaiApiOrganization)
#         self.orgEdit.setEchoMode(QLineEdit.Password)
        
#         # Add provider selection
#         self.providerBox = QComboBox()
#         for provider in ("openai", "anthropic"):
#             self.providerBox.addItem(provider)
#         self.providerBox.setCurrentText(getattr(config, "provider", "openai"))
#         self.providerBox.currentTextChanged.connect(self.updateModelChoices)
        
#         self.apiModelBox = QComboBox()
#         self.updateModelChoices(self.providerBox.currentText())
        
#         self.functionCallingBox = QComboBox()
#         initialIndex = 0
#         index = 0
#         for key in ("auto", "none"):
#             self.functionCallingBox.addItem(key)
#             if key == config.chatGPTApiFunctionCall:
#                 initialIndex = index
#             index += 1
#         self.functionCallingBox.setCurrentIndex(initialIndex)
#         self.loadingInternetSearchesBox = QComboBox()
#         initialIndex = 0
#         index = 0
#         for key in ("always", "auto", "none"):
#             self.loadingInternetSearchesBox.addItem(key)
#             if key == config.loadingInternetSearches:
#                 initialIndex = index
#             index += 1
#         self.loadingInternetSearchesBox.setCurrentIndex(initialIndex)
#         self.maxTokenEdit = QLineEdit(str(config.chatGPTApiMaxTokens))
#         self.maxTokenEdit.setToolTip("The maximum number of tokens to generate in the completion.\nThe token count of your prompt plus max_tokens cannot exceed the model's context length. Most models have a context length of 2048 tokens (except for the newest models, which support 4096).")
#         self.maxInternetSearchResults = QLineEdit(str(config.maximumInternetSearchResults))
#         self.maxInternetSearchResults.setToolTip("The maximum number of internet search response to be included.")
#         #self.includeInternetSearches = QCheckBox(config.thisTranslation["include"])
#         #self.includeInternetSearches.setToolTip("Include latest internet search results")
#         #self.includeInternetSearches.setCheckState(Qt.Checked if config.includeDuckDuckGoSearchResults else Qt.Unchecked)
#         #self.includeDuckDuckGoSearchResults = config.includeDuckDuckGoSearchResults
#         self.autoScrollingCheckBox = QCheckBox(config.thisTranslation["enable"])
#         self.autoScrollingCheckBox.setToolTip("Auto-scroll display as responses are received")
#         self.autoScrollingCheckBox.setCheckState(Qt.Checked if config.chatGPTApiAutoScrolling else Qt.Unchecked)
#         self.chatGPTApiAutoScrolling = config.chatGPTApiAutoScrolling
#         self.chatAfterFunctionCalledCheckBox = QCheckBox(config.thisTranslation["enable"])
#         self.chatAfterFunctionCalledCheckBox.setToolTip("Automatically generate next chat response after a function is called")
#         self.chatAfterFunctionCalledCheckBox.setCheckState(Qt.Checked if config.chatAfterFunctionCalled else Qt.Unchecked)
#         self.chatAfterFunctionCalled = config.chatAfterFunctionCalled
#         self.runPythonScriptGloballyCheckBox = QCheckBox(config.thisTranslation["enable"])
#         self.runPythonScriptGloballyCheckBox.setToolTip("Run user python script in global scope")
#         self.runPythonScriptGloballyCheckBox.setCheckState(Qt.Checked if config.runPythonScriptGlobally else Qt.Unchecked)
#         self.runPythonScriptGlobally = config.runPythonScriptGlobally
#         self.contextEdit = QLineEdit(config.chatGPTApiContext)
#         firstInputOnly = config.thisTranslation["firstInputOnly"]
#         allInputs = config.thisTranslation["allInputs"]
#         self.applyContextIn = QComboBox()
#         self.applyContextIn.addItems([firstInputOnly, allInputs])
#         self.applyContextIn.setCurrentIndex(1 if config.chatGPTApiContextInAllInputs else 0)
#         self.predefinedContextBox = QComboBox()
#         initialIndex = 0
#         index = 0
#         for key, value in config.predefinedContexts.items():
#             self.predefinedContextBox.addItem(key)
#             self.predefinedContextBox.setItemData(self.predefinedContextBox.count()-1, value, role=Qt.ToolTipRole)
#             if key == config.chatGPTApiPredefinedContext:
#                 initialIndex = index
#             index += 1
#         self.predefinedContextBox.currentIndexChanged.connect(self.predefinedContextBoxChanged)
#         self.predefinedContextBox.setCurrentIndex(initialIndex)
#         # set availability of self.contextEdit in case there is no index changed
#         self.contextEdit.setDisabled(True) if not initialIndex == 1 else self.contextEdit.setEnabled(True)
#         buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
#         buttonBox.accepted.connect(self.accept)
#         buttonBox.rejected.connect(self.reject)

#         layout = QFormLayout()
#         # https://platform.openai.com/account/api-keys
#         chatAfterFunctionCalled = config.thisTranslation["chatAfterFunctionCalled"]
#         runPythonScriptGlobally = config.thisTranslation["runPythonScriptGlobally"]
#         autoScroll = config.thisTranslation["autoScroll"]
#         predefinedContext = config.thisTranslation["predefinedContext"]
#         context = config.thisTranslation["chatContext"]
#         applyContext = config.thisTranslation["applyContext"]
#         latestOnlineSearchResults = config.thisTranslation["latestOnlineSearchResults"]
#         maximumOnlineSearchResults = config.thisTranslation["maximumOnlineSearchResults"]
#         required = config.thisTranslation["required"]
#         optional = config.thisTranslation["optional"]
#          # Add provider selection to layout
#         layout.addRow(f"Provider [{required}]:", self.providerBox)
#         layout.addRow(f"OpenAI API Key [{required}]:", self.apiKeyEdit)
#         layout.addRow(f"Organization ID [{optional}]:", self.orgEdit)
#         layout.addRow(f"API Model [{required}]:", self.apiModelBox)
#         layout.addRow(f"Max Token [{required}]:", self.maxTokenEdit)
#         layout.addRow(f"Function Calling [{optional}]:", self.functionCallingBox)
#         layout.addRow(f"{chatAfterFunctionCalled} [{optional}]:", self.chatAfterFunctionCalledCheckBox)
#         layout.addRow(f"{predefinedContext} [{optional}]:", self.predefinedContextBox)
#         layout.addRow(f"{context} [{optional}]:", self.contextEdit)
#         layout.addRow(f"{applyContext} [{optional}]:", self.applyContextIn)
#         layout.addRow(f"{latestOnlineSearchResults} [{optional}]:", self.loadingInternetSearchesBox)
#         layout.addRow(f"{maximumOnlineSearchResults} [{optional}]:", self.maxInternetSearchResults)
#         layout.addRow(f"{autoScroll} [{optional}]:", self.autoScrollingCheckBox)
#         layout.addRow(f"{runPythonScriptGlobally} [{optional}]:", self.runPythonScriptGloballyCheckBox)
#         layout.addWidget(buttonBox)
#         self.autoScrollingCheckBox.stateChanged.connect(self.toggleAutoScrollingCheckBox)
#         self.chatAfterFunctionCalledCheckBox.stateChanged.connect(self.toggleChatAfterFunctionCalled)
#         self.runPythonScriptGloballyCheckBox.stateChanged.connect(self.toggleRunPythonScriptGlobally)
#         self.functionCallingBox.currentIndexChanged.connect(self.functionCallingBoxChanged)
#         self.loadingInternetSearchesBox.currentIndexChanged.connect(self.loadingInternetSearchesBoxChanged)

#         self.setLayout(layout)

#     def updateModelChoices(self, provider):
#         self.apiModelBox.clear()
#         if provider == "openai":
#             models = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"]
#         else:  # anthropic
#             models = ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
            
#         initialIndex = 0
#         index = 0
#         for key in models:
#             self.apiModelBox.addItem(key)
#             if key == config.chatGPTApiModel:
#                 initialIndex = index
#             index += 1
#         self.apiModelBox.setCurrentIndex(initialIndex)

#     def provider(self):
#         return self.providerBox.currentText()

#     def api_key(self):
#         return self.apiKeyEdit.text().strip()

#     def org(self):
#         return self.orgEdit.text().strip()

#     def context(self):
#         return self.contextEdit.text().strip()

#     def contextInAllInputs(self):
#         return True if self.applyContextIn.currentIndex() == 1 else False

#     def predefinedContextBoxChanged(self, index):
#         self.contextEdit.setDisabled(True) if not index == 1 else self.contextEdit.setEnabled(True)

#     def predefinedContext(self):
#         return self.predefinedContextBox.currentText()
#         #return self.predefinedContextBox.currentData(Qt.ToolTipRole)

#     def apiModel(self):
#         #return "gpt-3.5-turbo"
#         return self.apiModelBox.currentText()

#     def functionCalling(self):
#         return self.functionCallingBox.currentText()

#     def max_token(self):
#         return self.maxTokenEdit.text().strip()

#     def enable_auto_scrolling(self):
#         return self.chatGPTApiAutoScrolling

#     def toggleAutoScrollingCheckBox(self, state):
#         self.chatGPTApiAutoScrolling = True if state else False

#     def enable_chatAfterFunctionCalled(self):
#         return self.chatAfterFunctionCalled

#     def toggleChatAfterFunctionCalled(self, state):
#         self.chatAfterFunctionCalled = True if state else False

#     def enable_runPythonScriptGlobally(self):
#         return self.runPythonScriptGlobally

#     def toggleRunPythonScriptGlobally(self, state):
#         self.runPythonScriptGlobally = True if state else False

#     def functionCallingBoxChanged(self):
#         if self.functionCallingBox.currentText() == "none" and self.loadingInternetSearchesBox.currentText() == "auto":
#             self.loadingInternetSearchesBox.setCurrentText("none")

#     def loadingInternetSearches(self):
#         return self.loadingInternetSearchesBox.currentText()

#     def loadingInternetSearchesBoxChanged(self, _):
#         if self.loadingInternetSearchesBox.currentText() == "auto":
#             self.functionCallingBox.setCurrentText("auto")

#     def max_token(self):
#         return self.maxTokenEdit.text().strip()

#     def max_internet_search_results(self):
#         return self.maxInternetSearchResults.text().strip()
    
class ApiSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(config.thisTranslation["settings"])

        # Add provider selection
        self.providerBox = QComboBox()
        for provider in ("anthropic"):
            self.providerBox.addItem(provider)
        self.providerBox.setCurrentText(getattr(config, "provider", "anthropic"))
        self.providerBox.currentTextChanged.connect(self.updateModelChoices)
        
        # Initialize API key from config
        self.apiKeyEdit = QLineEdit(getattr(config, "anthropicApiKey", ""))
        self.apiKeyEdit.setEchoMode(QLineEdit.Password)
        
        self.apiModelBox = QComboBox()
        self.updateModelChoices(self.providerBox.currentText())
        
        self.autoScrollingCheckBox = QCheckBox(config.thisTranslation["enable"])
        self.autoScrollingCheckBox.setToolTip("Auto-scroll display as responses are received")
        self.autoScrollingCheckBox.setCheckState(Qt.Checked if config.chatGPTApiAutoScrolling else Qt.Unchecked)
        self.chatGPTApiAutoScrolling = config.chatGPTApiAutoScrolling

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        layout = QFormLayout()
        required = config.thisTranslation["required"]
        optional = config.thisTranslation["optional"]
        autoScroll = config.thisTranslation["autoScroll"]

        layout.addRow(f"Provider [{required}]:", self.providerBox)
        layout.addRow(f"API Key [{required}]:", self.apiKeyEdit)
        layout.addRow(f"API Model [{required}]:", self.apiModelBox)
        layout.addRow(f"{autoScroll} [{optional}]:", self.autoScrollingCheckBox)
        layout.addWidget(buttonBox)
        
        self.autoScrollingCheckBox.stateChanged.connect(self.toggleAutoScrollingCheckBox)
        self.setLayout(layout)

    def updateModelChoices(self, provider):
        self.apiModelBox.clear()
        if provider == "anthropic":
            models = ["claude-3-5-sonnet-20241022"]
        else:
            models = ["claude-3-5-sonnet-20241022"]
            
        initialIndex = 0
        index = 0
        for key in models:
            self.apiModelBox.addItem(key)
            if key == config.chatGPTApiModel:
                initialIndex = index
            index += 1
        self.apiModelBox.setCurrentIndex(initialIndex)

    def provider(self):
        return self.providerBox.currentText()

    def api_key(self):
        return self.apiKeyEdit.text().strip()

    def apiModel(self):
        return self.apiModelBox.currentText()

    def enable_auto_scrolling(self):
        return self.chatGPTApiAutoScrolling

    def toggleAutoScrollingCheckBox(self, state):
        self.chatGPTApiAutoScrolling = True if state else False

class Database:
    def __init__(self, filePath=""):
        def regexp(expr, item):
            reg = re.compile(expr, flags=re.IGNORECASE)
            return reg.search(item) is not None
        
        # print("about to set working directory - this will break on built version")
        # this completely breaks on built version
        # set working directory - this is legacy nonsense that database uses below, but I think won't work
        # thisFile = os.path.realpath(__file__)
        # wd = os.path.dirname(thisFile)
        # if os.getcwd() != wd:
        #     os.chdir(wd)
        wd = user_data_dir(config.getAppName())
        print("finished setting working directory", wd)
        # Determine default file path:
        # 1. Use last chat database if it exists and is valid
        # 2. Otherwise use "default.chat" in the chats directory
        if (config.chatGPTApiLastChatDatabase and 
            os.path.isfile(config.chatGPTApiLastChatDatabase)):
            defaultFilePath = config.chatGPTApiLastChatDatabase
            print("using last chat database", defaultFilePath)
        else:
            defaultFilePath = os.path.join(config.getChatsPath(), "default.chat")
            print("using default chat database", defaultFilePath)
            
        self.filePath = filePath if filePath else defaultFilePath
        self.connection = sqlite3.connect(self.filePath)
        self.connection.create_function("REGEXP", 2, regexp)
        self.cursor = self.connection.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS data (id TEXT PRIMARY KEY, title TEXT, content TEXT)')
        self.connection.commit()

    def insert(self, id, title, content):
        self.cursor.execute('SELECT * FROM data WHERE id = ?', (id,))
        existing_data = self.cursor.fetchone()
        if existing_data:
            if existing_data[1] == title and existing_data[2] == content:
                return
            else:
                self.cursor.execute('UPDATE data SET title = ?, content = ? WHERE id = ?', (title, content, id))
                self.connection.commit()
        else:
            self.cursor.execute('INSERT INTO data (id, title, content) VALUES (?, ?, ?)', (id, title, content))
            self.connection.commit()

    def search(self, title, content):
        if config.regexpSearchEnabled:
            # with regular expression
            self.cursor.execute('SELECT * FROM data WHERE title REGEXP ? AND content REGEXP ?', (title, content))
        else:
            # without regular expression
            self.cursor.execute('SELECT * FROM data WHERE title LIKE ? AND content LIKE ?', ('%{}%'.format(title), '%{}%'.format(content)))
        return self.cursor.fetchall()

    def delete(self, id):
        self.cursor.execute('DELETE FROM data WHERE id = ?', (id,))
        self.connection.commit()

    def clear(self):
        self.cursor.execute('DELETE FROM data')
        self.connection.commit()

