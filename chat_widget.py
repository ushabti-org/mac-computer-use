import config
from configDefault import *

from core_components import ApiSettingsDialog, Database, SpeechRecognitionThread
from PySide6.QtWidgets import QWidget, QFileDialog
import shutil, glob, subprocess, traceback
import urllib.parse
from io import StringIO
from functools import partial

import re, openai, tiktoken, webbrowser, sys, os
from gtts import gTTS
try:
    from pocketsphinx import LiveSpeech, get_model_path
    isPocketsphinxInstalled = True
except:
    isPocketsphinxInstalled = False

from datetime import datetime
# import util.Worker
# from util.Worker import ChatGPTResponse, OpenAIImage

### COMPLETELY BRAINDEAD DROP-IN OF WORKER
### COMPLETELY BRAINDEAD DROP-IN OF WORKER
### COMPLETELY BRAINDEAD DROP-IN OF WORKER
import config

import sys, traceback, openai, os, json, traceback, re, textwrap
if config.qtLibrary == "pyside6":
    from PySide6.QtCore import QRunnable, Slot, Signal, QObject, QThreadPool
else:
    from qtpy.QtCore import QRunnable, Slot, Signal, QObject, QThreadPool



class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(str)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs["progress_callback"] = self.signals.progress

    @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # assign a reference to this current thread
        #config.workerThread = QThread.currentThread()

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class ChatGPTResponse:

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.threadpool = QThreadPool()

    def fineTunePythonCode(self, code):
        insert_string = "import config\nconfig.pythonFunctionResponse = "
        code = re.sub("^!(.*?)$", r"import os\nos.system(\1)", code, flags=re.M)
        if "\n" in code:
            substrings = code.rsplit("\n", 1)
            lastLine = re.sub("print\((.*)\)", r"\1", substrings[-1])
            code = code if lastLine.startswith(" ") else f"{substrings[0]}\n{insert_string}{lastLine}"
        else:
            code = f"{insert_string}{code}"
        return code

    def getFunctionResponse(self, response_message, function_name):
        if function_name == "python":
            config.pythonFunctionResponse = ""
            python_code = textwrap.dedent(response_message["function_call"]["arguments"])
            refinedCode = self.fineTunePythonCode(python_code)

            print("--------------------")
            print(f"running python code ...")
            if config.developer or config.codeDisplay:
                print("```")
                print(python_code)
                print("```")
            print("--------------------")

            try:
                exec(refinedCode, globals())
                function_response = str(config.pythonFunctionResponse)
            except:
                function_response = python_code
            info = {"information": function_response}
            function_response = json.dumps(info)
        else:
            fuction_to_call = config.chatGPTApiAvailableFunctions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            function_response = fuction_to_call(function_args)
        return function_response

    def getStreamFunctionResponseMessage(self, completion, function_name):
        function_arguments = ""
        for event in completion:
            delta = event["choices"][0]["delta"]
            if delta and delta.get("function_call"):
                function_arguments += delta["function_call"]["arguments"]
        return {
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": function_name,
                "arguments": function_arguments,
            }
        }

    def runCompletion(self, thisMessage, progress_callback):
        self.functionJustCalled = False
        def runThisCompletion(thisThisMessage):
            if config.chatGPTApiFunctionSignatures and not self.functionJustCalled:
                return openai.ChatCompletion.create(
                    model=config.chatGPTApiModel,
                    messages=thisThisMessage,
                    n=1,
                    temperature=config.chatGPTApiTemperature,
                    max_tokens=config.chatGPTApiMaxTokens,
                    functions=config.chatGPTApiFunctionSignatures,
                    function_call=config.chatGPTApiFunctionCall,
                    stream=True,
                )
            return openai.ChatCompletion.create(
                model=config.chatGPTApiModel,
                messages=thisThisMessage,
                n=1,
                temperature=config.chatGPTApiTemperature,
                max_tokens=config.chatGPTApiMaxTokens,
                stream=True,
            )

        while True:
            completion = runThisCompletion(thisMessage)
            function_name = ""
            try:
                # consume the first delta
                for event in completion:
                    delta = event["choices"][0]["delta"]
                    # Check if a function is called
                    if not delta.get("function_call"):
                        self.functionJustCalled = True
                    elif "name" in delta["function_call"]:
                        function_name = delta["function_call"]["name"]
                    # check the first delta is enough
                    break
                # Continue only when a function is called
                if self.functionJustCalled:
                    break

                # get stream function response message
                response_message = self.getStreamFunctionResponseMessage(completion, function_name)

                # get function response
                function_response = self.getFunctionResponse(response_message, function_name)

                # process function response
                # send the info on the function call and function response to GPT
                thisMessage.append(response_message) # extend conversation with assistant's reply
                thisMessage.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response

                self.functionJustCalled = True

                if not config.chatAfterFunctionCalled:
                    progress_callback.emit("\n\n~~~ ")
                    progress_callback.emit(function_response)
                    return None
            except:
                self.showErrors()
                break

        return completion

    def showErrors(self):
        if config.developer:
            print(traceback.format_exc())

    def getResponse(self, messages, progress_callback, functionJustCalled=False):
        responses = ""
        if config.loadingInternetSearches == "always" and not functionJustCalled:
            #print("loading internet searches ...")
            try:
                completion = openai.ChatCompletion.create(
                    model=config.chatGPTApiModel,
                    messages=messages,
                    max_tokens=config.chatGPTApiMaxTokens,
                    temperature=config.chatGPTApiTemperature,
                    n=1,
                    functions=config.integrate_google_searches_signature,
                    function_call={"name": "integrate_google_searches"},
                )
                response_message = completion["choices"][0]["message"]
                if response_message.get("function_call"):
                    function_args = json.loads(response_message["function_call"]["arguments"])
                    fuction_to_call = config.chatGPTApiAvailableFunctions.get("integrate_google_searches")
                    function_response = fuction_to_call(function_args)
                    messages.append(response_message) # extend conversation with assistant's reply
                    messages.append(
                        {
                            "role": "function",
                            "name": "integrate_google_searches",
                            "content": function_response,
                        }
                    )
            except:
                print("Unable to load internet resources.")
        try:
            if config.chatGPTApiNoOfChoices == 1:
                completion = self.runCompletion(messages, progress_callback)
                if completion is not None:
                    progress_callback.emit("\n\n~~~ ")
                    for event in completion:
                        # stop generating response
                        stop_file = ".stop_chatgpt"
                        if os.path.isfile(stop_file):
                            os.remove(stop_file)
                            break                                    
                        # RETRIEVE THE TEXT FROM THE RESPONSE
                        event_text = event["choices"][0]["delta"] # EVENT DELTA RESPONSE
                        progress = event_text.get("content", "") # RETRIEVE CONTENT
                        # STREAM THE ANSWER
                        progress_callback.emit(progress)
            else:
                if config.chatGPTApiFunctionSignatures:
                    completion = openai.ChatCompletion.create(
                        model=config.chatGPTApiModel,
                        messages=messages,
                        max_tokens=config.chatGPTApiMaxTokens,
                        temperature=0.0 if config.chatGPTApiPredefinedContext == "Execute Python Code" else config.chatGPTApiTemperature,
                        n=config.chatGPTApiNoOfChoices,
                        functions=config.chatGPTApiFunctionSignatures,
                        function_call={"name": "run_python"} if config.chatGPTApiPredefinedContext == "Execute Python Code" else config.chatGPTApiFunctionCall,
                    )
                else:
                    completion = openai.ChatCompletion.create(
                        model=config.chatGPTApiModel,
                        messages=messages,
                        max_tokens=config.chatGPTApiMaxTokens,
                        temperature=config.chatGPTApiTemperature,
                        n=config.chatGPTApiNoOfChoices,
                    )

                response_message = completion["choices"][0]["message"]
                if response_message.get("function_call"):
                    function_name = response_message["function_call"]["name"]
                    if function_name == "python":
                        config.pythonFunctionResponse = ""
                        function_args = response_message["function_call"]["arguments"]
                        insert_string = "import config\nconfig.pythonFunctionResponse = "
                        if "\n" in function_args:
                            substrings = function_args.rsplit("\n", 1)
                            new_function_args = f"{substrings[0]}\n{insert_string}{substrings[-1]}"
                        else:
                            new_function_args = f"{insert_string}{function_args}"
                        try:
                            exec(new_function_args, globals())
                            function_response = str(config.pythonFunctionResponse)
                        except:
                            function_response = function_args
                        info = {"information": function_response}
                        function_response = json.dumps(info)
                    else:
                        #if not function_name in config.chatGPTApiAvailableFunctions:
                        #    print("unexpected function name: ", function_name)
                        fuction_to_call = config.chatGPTApiAvailableFunctions.get(function_name, "integrate_google_searches")
                        try:
                            function_args = json.loads(response_message["function_call"]["arguments"])
                        except:
                            function_args = response_message["function_call"]["arguments"]
                            if function_name == "integrate_google_searches":
                                function_args = {"keywords": function_args}
                        function_response = fuction_to_call(function_args)
                    
                    # check function response
                    # print("Got this function response:", function_response)

                    # process function response
                    # send the info on the function call and function response to GPT
                    messages.append(response_message) # extend conversation with assistant's reply
                    messages.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": function_response,
                        }
                    )  # extend conversation with function response
                    if config.chatAfterFunctionCalled:
                        return self.getResponse(messages, progress_callback, functionJustCalled=True)
                    else:
                        responses += f"{function_response}\n\n"

                for index, choice in enumerate(completion.choices):
                    chat_response = choice.message.content
                    if chat_response:
                        if len(completion.choices) > 1:
                            if index > 0:
                                responses += "\n"
                            responses += f"~~~ Response {(index+1)}:\n"
                        responses += f"{chat_response}\n\n"
        # error codes: https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        except openai.error.APIError as e:
            #Handle API error here, e.g. retry or log
            return f"OpenAI API returned an API Error: {e}"
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            return f"Failed to connect to OpenAI API: {e}"
        except openai.error.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            return f"OpenAI API request exceeded rate limit: {e}"
        except:
            #traceback.print_exc()
            responses = traceback.format_exc()
        return responses

    def workOnGetResponse(self, messages):
        # Pass the function to execute
        worker = Worker(self.getResponse, messages) # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.parent.processResponse)
        worker.signals.progress.connect(self.parent.printStream)
        # Connection
        #worker.signals.finished.connect(None)
        # Execute
        self.threadpool.start(worker)


class OpenAIImage:

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.threadpool = QThreadPool()

    def getResponse(self, prompt, progress_callback=None):
        try:
            #https://platform.openai.com/docs/guides/images/introduction
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1024",
            )
            return response['data'][0]['url']
        # error codes: https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        except openai.error.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
        except openai.error.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
        except:
            traceback.print_exc()
        return ""

    def workOnGetResponse(self, prompt):
        # Pass the function to execute
        worker = Worker(self.getResponse, prompt) # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.parent.displayImage)
        # Connection
        #worker.signals.finished.connect(None)
        # Execute
        self.threadpool.start(worker)
### END COMPLETELY BRAINDEAD DROP-IN OF WORKER ###
### END COMPLETELY BRAINDEAD DROP-IN OF WORKER ###
### END COMPLETELY BRAINDEAD DROP-IN OF WORKER ###


if config.qtLibrary == "pyside6":
    from PySide6.QtPrintSupport import QPrinter, QPrintDialog
    from PySide6.QtCore import Qt, QThread, Signal, QRegularExpression
    from PySide6.QtGui import QStandardItemModel, QStandardItem, QGuiApplication, QAction, QIcon, QFontMetrics, QTextDocument, QClipboard, QImage
    from PySide6.QtWidgets import QCompleter, QMenu, QSystemTrayIcon, QApplication, QMainWindow, QWidget, QDialog, QFileDialog, QDialogButtonBox, QFormLayout, QLabel, QMessageBox, QCheckBox, QPlainTextEdit, QProgressBar, QPushButton, QListView, QHBoxLayout, QVBoxLayout, QLineEdit, QSplitter, QComboBox
    from PySide6.QtTest import QTest
else:
    from qtpy.QtPrintSupport import QPrinter, QPrintDialog
    from qtpy.QtCore import Qt, QThread, Signal, QRegularExpression
    from qtpy.QtGui import QStandardItemModel, QStandardItem, QGuiApplication, QIcon, QFontMetrics, QTextDocument, QClipboard
    from qtpy.QtWidgets import QCompleter, QMenu, QSystemTrayIcon, QApplication, QMainWindow, QAction, QWidget, QDialog, QFileDialog, QDialogButtonBox, QFormLayout, QLabel, QMessageBox, QCheckBox, QPlainTextEdit, QProgressBar, QPushButton, QListView, QHBoxLayout, QVBoxLayout, QLineEdit, QSplitter, QComboBox
    from qtpy.QtTest import QTest

import importlib  
from contextlib import contextmanager
loop = importlib.import_module("computer_use_demo.loop")
sampling_loop = loop.sampling_loop  # Access the function directly from the imported module
# tools = importlib.import_module("computer-use-demo.computer_use_demo.tools")
# from tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

from enum import StrEnum
import asyncio
import trio
import httpx

# TODO this seems wrong - need to read this from somewhere?
# anthropic tools get initialized based on these
# os.environ["HEIGHT"] = "768"
# os.environ["WIDTH"] = "1024" 
# os.environ["DISPLAY_NUM"] = "1"

global g_messages, g_responses, g_api_key, g_tools, g_custom_system_prompt, g_model, g_provider, g_only_n_most_recent_images
g_messages = []
g_responses = {}
g_api_key = getattr(config, "anthropicApiKey", "")
g_tools = {}
g_custom_system_prompt = ""
g_model = "claude-3-5-sonnet-20241022"
g_provider = "anthropic"
g_only_n_most_recent_images = 3
g_in_sampling_loop = False

class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"

def _api_response_callback(
    request: httpx.Request,
    response: httpx.Response | object | None,
    error: Exception | None,
    tab,
    # tab: DeltaGenerator,
    response_state: dict[str, tuple[httpx.Request, httpx.Response | object | None]],
    chat_gpt_api_widget,
):
    """
    Handle an API response by storing it to state and rendering it.
    """
    print("_api_response_callback TODO")
    print(f"request: {request}")
    print(f"response: {response}")
    print(f"error: {error}")
    print(f"tab: {tab}")
    print(f"response_state: {response_state}")
    # return;
    response_id = datetime.now().isoformat()
    response_state[response_id] = (request, response)
    if error:
        _render_error(error, chat_gpt_api_widget=chat_gpt_api_widget)
    _render_api_response(request, response, response_id, tab, chat_gpt_api_widget=chat_gpt_api_widget)

def _tool_output_callback(
    tool_output, 
    # tool_output: ToolResult, 
    tool_id: str, 
    tool_state,
    # tool_state: dict[str, ToolResult]
    chat_gpt_api_widget,
):
    """Handle a tool output by storing it to state and rendering it."""
    print("_tool_output_callback TODO")
    # Create a copy of tool_output for printing that masks the base64 image
    print_output = tool_output
    if hasattr(tool_output, 'base64_image') and tool_output.base64_image:
        print_output = tool_output.replace(base64_image='<image>')
    print(f"tool_output: {print_output}")
    print(f"tool_id: {tool_id}")
    # Create a copy of tool_state for printing with masked base64 images
    print_state = {}
    for k, v in tool_state.items():
        if hasattr(v, 'base64_image') and v.base64_image:
            print_state[k] = v.replace(base64_image='<image>')
        else:
            print_state[k] = v
    print(f"tool_state: {print_state}")
    tool_state[tool_id] = tool_output
    _render_message(Sender.TOOL, tool_output, chat_gpt_api_widget=chat_gpt_api_widget)

def _render_api_response(
    request: httpx.Request,
    response: httpx.Response | object | None,
    response_id: str,
    tab,
    # tab: DeltaGenerator,
    chat_gpt_api_widget,
):
    """Render an API response to a streamlit tab"""
    print("_render_api_response TODO")
    print(f"request: {request}")
    print(f"response: {response}")
    print(f"response_id: {response_id}")
    print(f"tab: {tab}")
    chat_gpt_api_widget.print(f"\n~~~ [API Response] [{response_id}] + request: {request} + response: {response}")
    QTest.qWait(0)
    return;
    with tab:
        with st.expander(f"Request/Response ({response_id})"):
            newline = "\n\n"
            st.markdown(
                f"`{request.method} {request.url}`{newline}{newline.join(f'`{k}: {v}`' for k, v in request.headers.items())}"
            )
            st.json(request.read().decode())
            st.markdown("---")
            if isinstance(response, httpx.Response):
                st.markdown(
                    f"`{response.status_code}`{newline}{newline.join(f'`{k}: {v}`' for k, v in response.headers.items())}"
                )
                st.json(response.text)
            else:
                st.write(response)


def _render_error(error: Exception, chat_gpt_api_widget):
    print("_render_error TODO")
    print(f"error: {error}")
    chat_gpt_api_widget.print(f"\n~~~ [Error] {error}")
    QTest.qWait(0)
    return;
    if isinstance(error, RateLimitError):
        body = "You have been rate limited."
        if retry_after := error.response.headers.get("retry-after"):
            body += f" **Retry after {str(timedelta(seconds=int(retry_after)))} (HH:MM:SS).** See our API [documentation](https://docs.anthropic.com/en/api/rate-limits) for more details."
        body += f"\n\n{error.message}"
    else:
        body = str(error)
        body += "\n\n**Traceback:**"
        lines = "\n".join(traceback.format_exception(error))
        body += f"\n\n```{lines}```"
    save_to_storage(f"error_{datetime.now().timestamp()}.md", body)
    st.error(f"**{error.__class__.__name__}**\n\n{body}", icon=":material/error:")


def _render_message(
    sender: Sender,
    message,
    # message: str | BetaContentBlockParam | ToolResult,
    chat_gpt_api_widget,
):
    """Convert input from the user or output from the agent to a streamlit message."""
    print("_render_message TODO")
    print(f"sender: {sender}")
    # Create a copy of message with masked base64 image for printing
    print_message = message
    if hasattr(message, 'base64_image') and message.base64_image:
        print_message = message.replace(base64_image='<image>')
    print(f"message: {print_message}")
    chat_gpt_api_widget.print(f"\n~~~ [{sender}] {print_message}")
    QTest.qWait(0)
    return;
    # streamlit's hotreloading breaks isinstance checks, so we need to check for class names
    is_tool_result = not isinstance(message, str | dict)
    if not message or (
        is_tool_result
        and st.session_state.hide_images
        and not hasattr(message, "error")
        and not hasattr(message, "output")
    ):
        return
    with st.chat_message(sender):
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                if message.__class__.__name__ == "CLIResult":
                    st.code(message.output)
                else:
                    st.markdown(message.output)
            if message.error:
                st.error(message.error)
            if message.base64_image and not st.session_state.hide_images:
                st.image(base64.b64decode(message.base64_image))
        elif isinstance(message, dict):
            if message["type"] == "text":
                st.write(message["text"])
            elif message["type"] == "tool_use":
                st.code(f'Tool Use: {message["name"]}\nInput: {message["input"]}')
            else:
                # only expected return types are text and tool_use
                raise Exception(f'Unexpected response type {message["type"]}')
        else:
            st.markdown(message)

@contextmanager
def track_sampling_loop():
    global g_in_sampling_loop
    g_in_sampling_loop = True
    yield
    g_in_sampling_loop = False


class ChatGPTAPI(QWidget):
    def __init__(self, parent):
        super().__init__()
        config.chatGPTApi = self
        self.parent = parent
        # required
        openai.api_key = os.environ["OPENAI_API_KEY"] = config.openaiApiKey
        # optional
        if config.openaiApiOrganization:
            openai.organization = config.openaiApiOrganization
        # set title
        self.setWindowTitle("mac-computer-use")
        # set variables
        self.setupVariables()
        # run plugins
        self.runPlugins()
        # setup interface
        self.setupUI()
        # load database
        self.loadData()
        # new entry at launch
        self.newData()

    def openDatabase(self):
        # Show a file dialog to get the file path to open
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Database", os.path.join(config.getChatsPath(), "default.chat"), "mac-computer-use Database (*.chat)", options=options)

        # If the user selects a file path, open the file
        self.database = Database(filePath)
        self.loadData()
        self.updateTitle(filePath)
        self.newData()

    def newDatabase(self, copyExistingDatabase=False):
        # Show a file dialog to get the file path to save
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filePath, _ = QFileDialog.getSaveFileName(self, "New Database", os.path.join(config.getChatsPath(), self.database.filePath if copyExistingDatabase else "new.chat"), "mac-computer-use Database (*.chat)", options=options)

        # If the user selects a file path, save the file
        if filePath:
            # make sure the file ends with ".chat"
            if not filePath.endswith(".chat"):
                filePath += ".chat"
            # ignore if copy currently opened database
            if copyExistingDatabase and os.path.abspath(filePath) == os.path.abspath(self.database.filePath):
                return
            # Check if the file already exists
            if os.path.exists(filePath):
                # Ask the user if they want to replace the existing file
                msgBox = QMessageBox()
                msgBox.setWindowTitle("Confirm overwrite")
                msgBox.setText(f"The file {filePath} already exists. Do you want to replace it?")
                msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msgBox.setDefaultButton(QMessageBox.No)
                if msgBox.exec() == QMessageBox.No:
                    return
                else:
                    os.remove(filePath)

            # create a new database
            if copyExistingDatabase:
                shutil.copy(self.database.filePath, filePath)
            self.database = Database(filePath)
            self.loadData()
            self.updateTitle(filePath)
            self.newData()

    def updateTitle(self, filePath=""):
        if not filePath:
            filePath = self.database.filePath
        config.chatGPTApiLastChatDatabase = filePath
        basename = os.path.basename(filePath)
        self.parent.setWindowTitle(f"mac-computer-use - {basename}")

    def setupVariables(self):
        self.busyLoading = False
        self.contentID = ""
        self.database = Database()
        self.updateTitle()
        self.data_list = []
        self.recognitionThread = SpeechRecognitionThread(self)
        self.recognitionThread.phrase_recognized.connect(self.onPhraseRecognized)

    def setupUI(self):
        layout000 = QHBoxLayout()
        self.setLayout(layout000)
        widgetLt = QWidget()
        layout000Lt = QVBoxLayout()
        widgetLt.setLayout(layout000Lt)
        widgetRt = QWidget()
        layout000Rt = QVBoxLayout()
        widgetRt.setLayout(layout000Rt)
        
        splitter = QSplitter(Qt.Horizontal, self)
        # splitter.addWidget(widgetLt) #hide left side 
        splitter.addWidget(widgetRt)
        layout000.addWidget(splitter)

        #widgets on the right
        self.searchInput = QLineEdit()
        self.searchInput.setClearButtonEnabled(True)
        self.replaceInput = QLineEdit()
        self.replaceInput.setClearButtonEnabled(True)
        self.userInput = QLineEdit()
        completer = QCompleter(config.inputSuggestions)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.userInput.setCompleter(completer)
        self.userInput.setFixedHeight(36)
        self.userInput.setPlaceholderText(config.thisTranslation["messageHere"])
        self.userInput.mousePressEvent = lambda _ : self.userInput.selectAll()
        self.userInput.setClearButtonEnabled(True)
        self.userInputMultiline = QPlainTextEdit()
        self.userInputMultiline.setPlaceholderText(config.thisTranslation["messageHere"])
        # Set fixed height to approximately 3 lines
        self.userInputMultiline.setFixedHeight(72)  # Assuming default line height is ~24px
        self.voiceCheckbox = QCheckBox(config.thisTranslation["voice"])
        self.voiceCheckbox.setToolTip(config.thisTranslation["voiceTyping"])
        self.voiceCheckbox.setCheckState(Qt.Unchecked)
        self.contentView = QPlainTextEdit()
        self.contentView.setReadOnly(True)
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 0) # Set the progress bar to use an indeterminate progress indicator
        apiKeyButton = QPushButton(config.thisTranslation["settings"])
        # self.multilineButton = QPushButton("+")
        # font_metrics = QFontMetrics(self.multilineButton.font())
        # text_rect = font_metrics.boundingRect(self.multilineButton.text())
        # button_width = text_rect.width() + 20
        # button_height = text_rect.height() + 10
        # self.multilineButton.setFixedSize(button_width, button_height)
        self.sendButton = QPushButton(config.thisTranslation["send"])
        # searchLabel = QLabel(config.thisTranslation["searchFor"])
        # replaceLabel = QLabel(config.thisTranslation["replaceWith"])
        # searchReplaceButton = QPushButton(config.thisTranslation["replace"])
        # searchReplaceButton.setToolTip(config.thisTranslation["replaceSelectedText"])
        # searchReplaceButtonAll = QPushButton(config.thisTranslation["all"])
        # searchReplaceButtonAll.setToolTip(config.thisTranslation["replaceAll"])
        self.apiModels = QComboBox()
        self.apiModels.addItems([config.thisTranslation["chat"], config.thisTranslation["image"], "browser", "python", "system"])
        self.apiModels.setCurrentIndex(0)
        self.apiModel = 0
        self.newButton = QPushButton(config.thisTranslation["new"])
        saveButton = QPushButton(config.thisTranslation["save"])
        # self.editableCheckbox = QCheckBox(config.thisTranslation["editable"])
        # self.editableCheckbox.setCheckState(Qt.Unchecked)
        #self.audioCheckbox = QCheckBox(config.thisTranslation["audio"])
        #self.audioCheckbox.setCheckState(Qt.Checked if config.chatGPTApiAudio else Qt.Unchecked)
        # self.choiceNumber = QComboBox()
        # self.choiceNumber.addItems([str(i) for i in range(1, 11)])
        # self.choiceNumber.setCurrentIndex((config.chatGPTApiNoOfChoices - 1))
        # self.fontSize = QComboBox()
        # self.fontSize.addItems([str(i) for i in range(1, 51)])
        # self.fontSize.setCurrentIndex((config.fontSize - 1))
        # self.temperature = QComboBox()
        # self.temperature.addItems([str(i/10) for i in range(0, 21)])
        # self.temperature.setCurrentIndex(int(config.chatGPTApiTemperature * 10))
        # temperatureLabel = QLabel(config.thisTranslation["temperature"])
        # temperatureLabel.setAlignment(Qt.AlignRight)
        # temperatureLabel.setToolTip("What sampling temperature to use, between 0 and 2. \nHigher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.")
        # choicesLabel = QLabel(config.thisTranslation["choices"])
        # choicesLabel.setAlignment(Qt.AlignRight)
        # choicesLabel.setToolTip("How many chat completion choices to generate for each input message.")
        # fontLabel = QLabel(config.thisTranslation["font"])
        # fontLabel.setAlignment(Qt.AlignRight)
        # fontLabel.setToolTip(config.thisTranslation["fontSize"])
        promptLayout = QHBoxLayout()
        userInputLayout = QVBoxLayout()
        userInputLayout.addWidget(self.userInput)
        # userInputLayout.addWidget(self.userInputMultiline)
        # Change default visibility - hide single line, show multiline
        # self.userInput.hide()
        self.userInputMultiline.hide()
        promptLayout.addLayout(userInputLayout)
        if isPocketsphinxInstalled:
            promptLayout.addWidget(self.voiceCheckbox)
        # promptLayout.addWidget(self.multilineButton)
        promptLayout.addWidget(self.sendButton)
        # promptLayout.addWidget(self.apiModels)
        layout000Rt.addWidget(self.contentView)
        layout000Rt.addLayout(promptLayout)
        layout000Rt.addWidget(self.progressBar)
        self.progressBar.hide()
        # searchReplaceLayout = QHBoxLayout()
        # searchReplaceLayout.addWidget(searchLabel)
        # searchReplaceLayout.addWidget(self.searchInput)
        # searchReplaceLayout.addWidget(replaceLabel)
        # searchReplaceLayout.addWidget(self.replaceInput)
        # searchReplaceLayout.addWidget(searchReplaceButton)
        # searchReplaceLayout.addWidget(searchReplaceButtonAll)
        # layout000Rt.addLayout(searchReplaceLayout)
        rtControlLayout = QHBoxLayout()
        rtControlLayout.addWidget(apiKeyButton)
        # rtControlLayout.addWidget(temperatureLabel)
        # rtControlLayout.addWidget(self.temperature)
        # rtControlLayout.addWidget(choicesLabel)
        # rtControlLayout.addWidget(self.choiceNumber)
        # rtControlLayout.addWidget(fontLabel)
        # rtControlLayout.addWidget(self.fontSize)
        # rtControlLayout.addWidget(self.editableCheckbox)
        #rtControlLayout.addWidget(self.audioCheckbox)
        rtButtonLayout = QHBoxLayout()
        rtButtonLayout.addWidget(self.newButton)
        rtButtonLayout.addWidget(saveButton)
        layout000Rt.addLayout(rtControlLayout)
        layout000Rt.addLayout(rtButtonLayout)
        
        #widgets on the left
        helpButton = QPushButton(config.thisTranslation["help"])
        searchTitleButton = QPushButton(config.thisTranslation["searchTitle"])
        searchContentButton = QPushButton(config.thisTranslation["searchContent"])
        self.searchTitle = QLineEdit()
        self.searchTitle.setClearButtonEnabled(True)
        self.searchTitle.setPlaceholderText(config.thisTranslation["searchTitleHere"])
        self.searchContent = QLineEdit()
        self.searchContent.setClearButtonEnabled(True)
        self.searchContent.setPlaceholderText(config.thisTranslation["searchContentHere"])
        self.listView = QListView()
        self.listModel = QStandardItemModel()
        self.listView.setModel(self.listModel)
        removeButton = QPushButton(config.thisTranslation["remove"])
        clearAllButton = QPushButton(config.thisTranslation["clearAll"])
        searchTitleLayout = QHBoxLayout()
        searchTitleLayout.addWidget(self.searchTitle)
        searchTitleLayout.addWidget(searchTitleButton)
        layout000Lt.addLayout(searchTitleLayout)
        searchContentLayout = QHBoxLayout()
        searchContentLayout.addWidget(self.searchContent)
        searchContentLayout.addWidget(searchContentButton)
        layout000Lt.addLayout(searchContentLayout)
        layout000Lt.addWidget(self.listView)
        ltButtonLayout = QHBoxLayout()
        ltButtonLayout.addWidget(removeButton)
        ltButtonLayout.addWidget(clearAllButton)
        layout000Lt.addLayout(ltButtonLayout)
        layout000Lt.addWidget(helpButton)
        
        # Connections
        self.userInput.returnPressed.connect(self.sendMessage)
        # self.userInputMultiline.returnPressed.connect(self.sendMessage) this never worked
        apiKeyButton.clicked.connect(self.showSettingsDialog)
        # self.multilineButton.clicked.connect(self.multilineButtonClicked)
        self.sendButton.clicked.connect(self.sendMessage)
        # just for async testing
        # self.sendButton.clicked.connect(self.parent.async_start)
        saveButton.clicked.connect(self.saveData)
        self.newButton.clicked.connect(self.newData)
        searchTitleButton.clicked.connect(self.searchData)
        searchContentButton.clicked.connect(self.searchData)
        self.searchTitle.textChanged.connect(self.searchData)
        self.searchContent.textChanged.connect(self.searchData)
        self.listView.clicked.connect(self.selectData)
        clearAllButton.clicked.connect(self.clearData)
        removeButton.clicked.connect(self.removeData)
        # self.editableCheckbox.stateChanged.connect(self.toggleEditable)
        #self.audioCheckbox.stateChanged.connect(self.toggleChatGPTApiAudio)
        # self.voiceCheckbox.stateChanged.connect(self.toggleVoiceTyping)
        # self.choiceNumber.currentIndexChanged.connect(self.updateChoiceNumber)
        self.apiModels.currentIndexChanged.connect(self.updateApiModel)
        # self.fontSize.currentIndexChanged.connect(self.setFontSize)
        # self.temperature.currentIndexChanged.connect(self.updateTemperature)
        # searchReplaceButton.clicked.connect(self.replaceSelectedText)
        # searchReplaceButtonAll.clicked.connect(self.searchReplaceAll)
        # self.searchInput.returnPressed.connect(self.searchChatContent)
        # self.replaceInput.returnPressed.connect(self.replaceSelectedText)

        self.setFontSize()
        self.updateSearchToolTips()

    def setFontSize(self, index=None):
        if index is not None:
            config.fontSize = index + 1
        # content view
        font = self.contentView.font()
        font.setPointSize(config.fontSize)
        self.contentView.setFont(font)
        # list view
        font = self.listView.font()
        font.setPointSize(config.fontSize)
        self.listView.setFont(font)

    def updateSearchToolTips(self):
        if config.regexpSearchEnabled:
            self.searchTitle.setToolTip(config.thisTranslation["matchingRegularExpression"])
            self.searchContent.setToolTip(config.thisTranslation["matchingRegularExpression"])
            self.searchInput.setToolTip(config.thisTranslation["matchingRegularExpression"])
        else:
            self.searchTitle.setToolTip("")
            self.searchContent.setToolTip("")
            self.searchInput.setToolTip("")

    def searchChatContent(self):
        search = QRegularExpression(self.searchInput.text()) if config.regexpSearchEnabled else self.searchInput.text()
        self.contentView.find(search)

    def replaceSelectedText(self):
        currentSelectedText = self.contentView.textCursor().selectedText()
        if not currentSelectedText == "":
            searchInput = self.searchInput.text()
            replaceInput = self.replaceInput.text()
            if searchInput:
                replace = re.sub(searchInput, replaceInput, currentSelectedText) if config.regexpSearchEnabled else currentSelectedText.replace(searchInput, replaceInput)
            else:
                replace = self.replaceInput.text()
            self.contentView.insertPlainText(replace)

    def searchReplaceAll(self):
        search = self.searchInput.text()
        if search:
            replace = self.replaceInput.text()
            content = self.contentView.toPlainText()
            newContent = re.sub(search, replace, content, flags=re.M) if config.regexpSearchEnabled else content.replace(search, replace)
            self.contentView.setPlainText(newContent)

    def multilineButtonClicked(self):
        print("multiline button")
        if self.userInput.isVisible():
            self.userInput.hide()
            self.userInputMultiline.setPlainText(self.userInput.text())
            self.userInputMultiline.show()
            self.multilineButton.setText("-")
        else:
            self.userInputMultiline.hide()
            self.userInput.setText(self.userInputMultiline.toPlainText())
            self.userInput.show()
            self.multilineButton.setText("+")
        self.setUserInputFocus()

    def setUserInputFocus(self):
        self.userInput.setFocus() if self.userInput.isVisible() else self.userInputMultiline.setFocus()

    def showSettingsDialog(self):
        dialog = ApiSettingsDialog(self)
        result = dialog.exec() if config.qtLibrary == "pyside6" else dialog.exec_()
        if result == QDialog.Accepted:
            global g_api_key
            # Store API key in both global var and config
            g_api_key = dialog.api_key()
            config.anthropicApiKey = g_api_key
            config.chatGPTApiAutoScrolling = dialog.enable_auto_scrolling()
            config.chatGPTApiModel = dialog.apiModel()
            config.provider = dialog.provider()
            self.newData()

    def updateApiModel(self, index):
        self.apiModel = index

    # def updateTemperature(self, index):
    #     config.chatGPTApiTemperature = float(index / 10)

    # def updateChoiceNumber(self, index):
    #     config.chatGPTApiNoOfChoices = index + 1

    def onPhraseRecognized(self, phrase):
        self.userInput.setText(f"{self.userInput.text()} {phrase}")

    # def toggleVoiceTyping(self, state):
    #     self.recognitionThread.start() if state else self.recognitionThread.stop()

    # def toggleEditable(self, state):
    #     self.contentView.setReadOnly(not state)

    def toggleChatGPTApiAudio(self, state):
        config.chatGPTApiAudio = state
        if not config.chatGPTApiAudio:
            self.closeMediaPlayer()

    def noTextSelection(self):
        self.displayMessage("This feature works on text selection. Select text first!")

    def validate_url(self, url):
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    # def webBrowse(self, userInput=""):
    #     if not userInput:
    #         userInput = self.contentView.textCursor().selectedText().strip()
    #     if not userInput:
    #         self.noTextSelection()
    #         return
    #     if self.validate_url(userInput):
    #         url = userInput
    #     else:
    #         userInput = urllib.parse.quote(userInput)
    #         url = f"https://www.google.com/search?q={userInput}"
    #     webbrowser.open(url)

    def displayText(self, text):
        self.saveData()
        self.newData()
        self.contentView.setPlainText(text)

    def runSystemCommand(self, command=""):
        if not command:
            command = self.contentView.textCursor().selectedText().strip()
        if command:
            command = repr(command)
            command = eval(command).replace("\u2029", "\n")
        else:
            self.noTextSelection()
            return
        
        # display output only, without error
        #output = subprocess.check_output(command, shell=True, text=True)
        #self.displayText(output)

        # display both output and error
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout  # Captured standard output
        error = result.stderr  # Captured standard error
        self.displayText(f"> {command}")
        self.contentView.appendPlainText(f"\n{output}")
        if error.strip():
            self.contentView.appendPlainText("\n# Error\n")
            self.contentView.appendPlainText(error)

    def runPythonCommand(self, command=""):
        if not command:
            command = self.contentView.textCursor().selectedText().strip()
        if command:
            command = repr(command)
            command = eval(command).replace("\u2029", "\n")
        else:
            self.noTextSelection()
            return

        # Store the original standard output
        original_stdout = sys.stdout
        # Create a StringIO object to capture the output
        output = StringIO()
        try:
            # Redirect the standard output to the StringIO object
            sys.stdout = output
            # Execute the Python string in global namespace
            try:
                exec(command, globals()) if config.runPythonScriptGlobally else exec(command)
                captured_output = output.getvalue()
            except:
                captured_output = traceback.format_exc()
            # Get the captured output
        finally:
            # Restore the original standard output
            sys.stdout = original_stdout

        # Display the captured output
        if captured_output.strip():
            self.displayText(captured_output)
        else:
            self.displayMessage("Done!")

    def removeData(self):
        index = self.listView.selectedIndexes()
        if not index:
            return
        confirm = QMessageBox.question(self, config.thisTranslation["remove"], config.thisTranslation["areyousure"], QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            item = index[0]
            data = item.data(Qt.UserRole)
            self.database.delete(data[0])
            self.loadData()
            self.newData()

    def clearData(self):
        confirm = QMessageBox.question(self, config.thisTranslation["clearAll"], config.thisTranslation["areyousure"], QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.database.clear()
            self.loadData()

    def saveData(self):
        text = self.contentView.toPlainText().strip()
        if text:
            lines = text.split("\n")
            if not self.contentID:
                self.contentID = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            title = re.sub("^>>> ", "", lines[0][:50])
            content = text
            self.database.insert(self.contentID, title, content)
            self.loadData()

    def loadData(self):
        # reverse the list, so that the latest is on the top
        self.data_list = self.database.search("", "")
        if self.data_list:
            self.data_list.reverse()
        self.listModel.clear()
        for data in self.data_list:
            item = QStandardItem(data[1])
            item.setToolTip(data[0])
            item.setData(data, Qt.UserRole)
            self.listModel.appendRow(item)

    def searchData(self):
        keyword1 = self.searchTitle.text().strip()
        keyword2 = self.searchContent.text().strip()
        self.data_list = self.database.search(keyword1, keyword2)
        self.listModel.clear()
        for data in self.data_list:
            item = QStandardItem(data[1])
            item.setData(data, Qt.UserRole)
            self.listModel.appendRow(item)

    def bibleChatAction(self, context=""):
        if context:
            config.chatGPTApiPredefinedContext = context
        currentSelectedText = self.contentView.textCursor().selectedText().strip()
        if currentSelectedText:
            self.newData()
            self.userInput.setText(currentSelectedText)
            self.sendMessage()


    def newData(self):
        if not self.busyLoading:
            self.contentID = ""
            self.contentView.setPlainText("" if openai.api_key else """OpenAI API Key is NOT Found!

Follow the following steps:
1) Register and get your OpenAI Key at https://platform.openai.com/account/api-keys
2) Click the "Settings" button below and enter your own OpenAI API key""")
            self.setUserInputFocus()

    def selectData(self, index):
        if not self.busyLoading:
            data = index.data(Qt.UserRole)
            self.contentID = data[0]
            content = data[2]
            self.contentView.setPlainText(content)
            self.setUserInputFocus()

    def printData(self):
        # Get the printer and print dialog
        printer = QPrinter()
        dialog = QPrintDialog(printer, self)

        # If the user clicked "OK" in the print dialog, print the text
        if dialog.exec() == QPrintDialog.Accepted:
            document = QTextDocument()
            document.setPlainText(self.contentView.toPlainText())
            document.print_(printer)

    def exportData(self):
        # Show a file dialog to get the file path to save
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filePath, _ = QFileDialog.getSaveFileName(self, "Export Chat Content", os.path.join(config.getChatsPath(), "chat.txt"), "Text Files (*.txt);;Python Files (*.py);;All Files (*)", options=options)

        # If the user selects a file path, save the file
        if filePath:
            with open(filePath, "w", encoding="utf-8") as fileObj:
                fileObj.write(self.contentView.toPlainText().strip())

    def openTextFileDialog(self):
        options = QFileDialog.Options()
        fileName, filtr = QFileDialog.getOpenFileName(self,
                                                      "Open Text File",
                                                      "Text File",
                                                      "Plain Text Files (*.txt);;Python Scripts (*.py);;All Files (*)",
                                                      "", options)
        if fileName:
            with open(fileName, "r", encoding="utf-8") as fileObj:
                self.displayText(fileObj.read())

    def displayMessage(self, message="", title="mac-computer-use"):
        QMessageBox.information(self, title, message)

    # The following method was modified from source:
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def num_tokens_from_messages(self, model=""):
        if not model:
            model = config.chatGPTApiModel
        userInput = self.userInput.text().strip()
        messages = self.getMessages(userInput)

        """Return the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        #encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            #print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            #print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        #return num_tokens
        self.displayMessage(message=f"{num_tokens} prompt tokens counted!")

    def getContext(self):
        if not config.chatGPTApiPredefinedContext in config.predefinedContexts:
            config.chatGPTApiPredefinedContext = "[none]"
        if config.chatGPTApiPredefinedContext == "[none]":
            # no context
            context = ""
        elif config.chatGPTApiPredefinedContext == "[custom]":
            # custom input in the settings dialog
            context = config.chatGPTApiContext
        else:
            # users can modify config.predefinedContexts via plugins
            context = config.predefinedContexts[config.chatGPTApiPredefinedContext]
            # change config for particular contexts
            if config.chatGPTApiPredefinedContext == "Execute Python Code":
                if config.chatGPTApiFunctionCall == "none":
                    config.chatGPTApiFunctionCall = "auto"
                if config.loadingInternetSearches == "always":
                    config.loadingInternetSearches = "auto"
        return context

    def getMessages(self, userInput):
        # system message
        systemMessage = "You're a kind helpful assistant."
        if config.chatGPTApiFunctionCall == "auto" and config.chatGPTApiFunctionSignatures:
            systemMessage += " Only use the functions you have been provided with."
        messages = [
            {"role": "system", "content": systemMessage}
        ]
        # predefine context
        context = self.getContext()
        # chat history
        history = self.contentView.toPlainText().strip()
        if history:
            if context and not config.chatGPTApiPredefinedContext == "Execute Python Code" and not config.chatGPTApiContextInAllInputs:
                messages.append({"role": "assistant", "content": context})
            if history.startswith(">>> "):
                history = history[4:]
            exchanges = [exchange for exchange in history.split("\n>>> ") if exchange.strip()]
            for exchange in exchanges:
                qa = exchange.split("\n~~~ ")
                for i, content in enumerate(qa):
                    if i == 0:
                        messages.append({"role": "user", "content": content.strip()})
                    else:
                        messages.append({"role": "assistant", "content": content.strip()})
        # customise chat context
        if context and (config.chatGPTApiPredefinedContext == "Execute Python Code" or (not history or (history and config.chatGPTApiContextInAllInputs))):
            #messages.append({"role": "assistant", "content": context})
            userInput = f"{context}\n{userInput}"
        # user input
        messages.append({"role": "user", "content": userInput})
        return messages

    def print(self, text):
        self.contentView.appendPlainText(f"\n{text}" if self.contentView.toPlainText() else text)
        self.contentView.setPlainText(re.sub("\n\n[\n]+?([^\n])", r"\n\n\1", self.contentView.toPlainText()))

    def printStream(self, text):
        # transform responses
        for t in config.chatGPTTransformers:
            text = t(text)
        self.contentView.setPlainText(self.contentView.toPlainText() + text)
        # no audio for streaming tokens
        #if config.chatGPTApiAudio:
        #    self.playAudio(text)
        # scroll to the bottom
        if config.chatGPTApiAutoScrolling:
            contentScrollBar = self.contentView.verticalScrollBar()
            contentScrollBar.setValue(contentScrollBar.maximum())

    def sendMessage(self):
        if self.userInputMultiline.isVisible():
            self.multilineButtonClicked()
        if self.apiModel == 0:
            self.getResponse()
        elif self.apiModel == 1:
            self.getImage()
        elif self.apiModel == 2:
            userInput = self.userInput.text().strip()
            # if userInput:
                # self.webBrowse(userInput)
        elif self.apiModel == 3:
            userInput = self.userInput.text().strip()
            if userInput:
                self.runPythonCommand(userInput)
        elif self.apiModel == 4:
            userInput = self.userInput.text().strip()
            if userInput:
                self.runSystemCommand(userInput)

    def getImage(self):
        if not self.progressBar.isVisible():
            userInput = self.userInput.text().strip()
            if userInput:
                self.userInput.setDisabled(True)
                self.progressBar.show() # show progress bar
                OpenAIImage(self).workOnGetResponse(userInput)

    def displayImage(self, imageUrl):
        if imageUrl:
            webbrowser.open(imageUrl)
            self.userInput.setEnabled(True)
            self.progressBar.hide()

    async def process_computer_use(self, userInput):
        # Use global variables instead of local declarations
        global g_messages, g_responses, g_tools, g_api_key, g_custom_system_prompt, g_model, g_provider, g_only_n_most_recent_images
        # TODO probably need to replace this print with a _render_message call
        self.print(f">>> {userInput}")
        message = {"role": "user", "content": userInput}
        g_messages.append(message)
        _render_message(Sender.USER, message=message, chat_gpt_api_widget=self)
        # await trio.sleep(0)
        # await asyncio.sleep(0)
        QTest.qWait(0)
        
        with track_sampling_loop():
            g_messages = await sampling_loop(
                system_prompt_suffix=g_custom_system_prompt,
                model=g_model,
                provider=g_provider,
                messages=g_messages,
                output_callback=partial(_render_message, Sender.BOT, chat_gpt_api_widget=self),
                tool_output_callback=partial(
                    _tool_output_callback, tool_state=g_tools, chat_gpt_api_widget=self
                ),
                api_response_callback=partial(
                    _api_response_callback,
                    tab=None,
                    # tab="http_logs_tab_lol_this_is_wrong_TODO",
                    # tab=http_logs,
                    response_state=g_responses,
                    chat_gpt_api_widget=self,
                ),
                api_key=g_api_key,
                only_n_most_recent_images=g_only_n_most_recent_images,
            )
            return g_messages

    def getResponse(self):
        print("in getResponse")
        if self.progressBar.isVisible() and config.chatGPTApiNoOfChoices == 1:
            stop_file = ".stop_chatgpt"
            if not os.path.isfile(stop_file):
                open(stop_file, "a", encoding="utf-8").close()
        elif not self.progressBar.isVisible():
            userInput = self.userInput.text().strip()
            if userInput:
                # Special case for computer use mode
                if getattr(config, "computerUseEnabled", False):
                    do_pfay_logic = True
                    # do_pfay_logic = False
                    if do_pfay_logic:
                        # Create event loop and run the async function
                        # note that we're returning messages but we could just as soon simply access the global g_messages
                        # messages = asyncio.run(process_computer_use())
                        # def task_callback(task):
                        #     messages = task.result()
                            
                        #     print(messages)
                        #     last_content = messages[-1]["content"]
                        #     print(last_content)
                        #     # TODO these won't work right if it's not 'text'
                        #     # TODO probably need to delete these, they are duplicating _render_message calls
                        #     if isinstance(last_content, list):
                        #         for content in last_content:
                        #             self.print(f"\n~~~ {content['text']}")
                        #     else:
                        #         if 'text' in last_content:
                        #             self.print(f"\n~~~ {last_content['text']}")
                        #         else:
                        #             self.print(f"\n~~~ {last_content}")

                        asyncio.run(self.process_computer_use(userInput))
                        # self.last_user_input = userInput # this is a real dumb argument pass
                        # self.parent.async_start()
                        # tsk = asyncio.get_running_loop().create_task(process_computer_use())
                        # tsk.add_done_callback(task_callback)
                        

                        # messages = await process_computer_use()
                        # event_loop = asyncio.get_event_loop()
                        # messages = event_loop.run_until_complete(process_computer_use())
                        # event_loop.run_until_complete(asyncio.ensure_future(process_computer_use()))

                        # async_trigger = QAction("test async", self)
                        # async_trigger.triggered.connect(lambda: asyncio.ensure_future(self.some_async_function()))
                        # text_selection_menu.addAction(async_trigger)

                    else:
                        self.print(f">>> {userInput}")
                        self.print(f"\n~~~ {userInput}")
                    self.userInput.setText("")
                    self.saveData()
                    return
                
                # Regular chat processing continues below...
                self.userInput.setDisabled(True)
                if config.chatGPTApiNoOfChoices == 1:
                    self.sendButton.setText(config.thisTranslation["stop"])
                    self.busyLoading = True
                    self.listView.setDisabled(True)
                    self.newButton.setDisabled(True)
                
                # Get provider and prepare messages
                provider = getattr(config, "provider", "openai")
                messages = self.getMessages(userInput)
                
                self.print(f">>> {userInput}")
                self.saveData()
                self.currentLoadingID = self.contentID
                self.currentLoadingContent = self.contentView.toPlainText().strip()
                self.progressBar.show()
                
                # Create appropriate client based on provider
                if provider == "anthropic":
                    try:
                        import anthropic
                        client = anthropic.Anthropic(api_key=config.openaiApiKey)
                        
                        # Convert messages format for Anthropic
                        anthropic_messages = []
                        for msg in messages:
                            if msg["role"] != "system":  # Anthropic doesn't use system messages
                                anthropic_messages.append({
                                    "role": msg["role"],
                                    "content": msg["content"]
                                })
                        
                        # Create message parameters
                        params = {
                            "model": config.chatGPTApiModel,
                            "max_tokens": config.chatGPTApiMaxTokens,
                            "messages": anthropic_messages,
                            "temperature": config.chatGPTApiTemperature,
                        }
                        
                        # Pass to worker with Anthropic client and params
                        ChatGPTResponse(self).workOnGetAnthropicResponse(client, params)
                        
                    except Exception as e:
                        self.print(f"\n~~~ Error: {str(e)}")
                        self.userInput.setEnabled(True)
                        self.progressBar.hide()
                        
                else:  # OpenAI
                    ChatGPTResponse(self).workOnGetResponse(messages)

    def fileNamesWithoutExtension(self, dir, ext):
        files = glob.glob(os.path.join(dir, "*.{0}".format(ext)))
        return sorted([file[len(dir)+1:-(len(ext)+1)] for file in files if os.path.isfile(file)])

    def execPythonFile(self, script):
        if config.developer:
            with open(script, 'r', encoding='utf8') as f:
                code = compile(f.read(), script, 'exec')
                exec(code, globals())
        else:
            try:
                with open(script, 'r', encoding='utf8') as f:
                    code = compile(f.read(), script, 'exec')
                    exec(code, globals())
            except:
                print("Failed to run '{0}'!".format(os.path.basename(script)))

    def runPlugins(self):
        # The following config values can be modified with plugins, to extend functionalities
        config.predefinedContexts = {
            "[none]": "",
            "[custom]": "",
        }
        config.inputSuggestions = []
        config.chatGPTTransformers = []
        config.chatGPTApiFunctionSignatures = []
        config.chatGPTApiAvailableFunctions = {}

        pluginFolder = os.path.join(os.getcwd(), "plugins")
        # always run 'integrate google searches'
        # internetSeraches = "integrate google searches"
        # script = os.path.join(pluginFolder, "{0}.py".format(internetSeraches))
        # self.execPythonFile(script)
        # for plugin in self.fileNamesWithoutExtension(pluginFolder, "py"):
        #     if not plugin == internetSeraches and not plugin in config.chatGPTPluginExcludeList:
        #         script = os.path.join(pluginFolder, "{0}.py".format(plugin))
        #         self.execPythonFile(script)
        # if internetSeraches in config.chatGPTPluginExcludeList:
        #     del config.chatGPTApiFunctionSignatures[0]

    def processResponse(self, responses):
        if responses:
            # reload the working content in case users change it during waiting for response
            self.contentID = self.currentLoadingID
            self.contentView.setPlainText(self.currentLoadingContent)
            self.currentLoadingID = self.currentLoadingContent = ""
            # transform responses
            for t in config.chatGPTTransformers:
                responses = t(responses)
            # update new reponses
            self.print(responses)
            # scroll to the bottom
            if config.chatGPTApiAutoScrolling:
                contentScrollBar = self.contentView.verticalScrollBar()
                contentScrollBar.setValue(contentScrollBar.maximum())
            #if not (responses.startswith("OpenAI API re") or responses.startswith("Failed to connect to OpenAI API:")) and config.chatGPTApiAudio:
            #        self.playAudio(responses)
        # empty user input
        self.userInput.setText("")
        # auto-save
        self.saveData()
        # hide progress bar
        self.userInput.setEnabled(True)
        if config.chatGPTApiNoOfChoices == 1:
            self.listView.setEnabled(True)
            self.newButton.setEnabled(True)
            self.busyLoading = False
        self.sendButton.setText(config.thisTranslation["send"])
        self.progressBar.hide()
        self.setUserInputFocus()

    def playAudio(self, responses):
        textList = [i.replace(">>>", "").strip() for i in responses.split("\n") if i.strip()]
        audioFiles = []
        for index, text in enumerate(textList):
            try:
                audioFile = os.path.abspath(os.path.join("temp", f"gtts_{index}.mp3"))
                if os.path.isfile(audioFile):
                    os.remove(audioFile)
                gTTS(text=text, lang=config.chatGPTApiAudioLanguage if config.chatGPTApiAudioLanguage else "en").save(audioFile)
                audioFiles.append(audioFile)
            except:
                pass
        if audioFiles:
            self.playAudioBibleFilePlayList(audioFiles)
    
    def playAudioBibleFilePlayList(self, files):
        pass

    def closeMediaPlayer(self):
        pass
