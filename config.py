# WARNING - if you add new functions here ensure they are ignored in aboutToQuit()

def getAppName():
    return "mac-computer-use"

def getConfigPath():
    import os
    from platformdirs import user_config_dir
    # with importlib.resources.path("config", "config.py") as configPath:
    # configPath = importlib.resources.path("config", "config.py")
    # configPath = os.path.join(wd, "config.py")
    appname = getAppName()
    configPath = os.path.join(user_config_dir(appname), "config.py")
    print("configPath: ", configPath)
    return configPath

def getChatsPath():
    import os
    from platformdirs import user_data_dir
    appname = getAppName()
    chatsPath = os.path.join(user_data_dir(appname), "chats")
    # print("chatsPath: ", chatsPath)
    return chatsPath

def setupConfig():
    import os
    import importlib.util as importlibutil
    from platformdirs import user_config_dir

    configPath = getConfigPath()
    # set working directory
    # thisFile = os.path.realpath(__file__)
    # wd = os.path.dirname(thisFile)
    # print("desired working directory: ", wd)
    # if os.getcwd() != wd:
    #     print("changing working directory from: ", os.getcwd())
    #     os.chdir(wd)
    if not os.path.isfile(configPath):
        print("creating config.py")
        os.makedirs(os.path.dirname(configPath), exist_ok=True)
        open(configPath, "a", encoding="utf-8").close()



    # thisFile = os.path.realpath(__file__)
    # appname = "mac-computer-use"
    # configPath = os.path.join(user_config_dir(appname), "config.py")
    # print("configPath: ", configPath)
    def load_config_module(config_path):
        spec = importlibutil.spec_from_file_location("config", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load the spec for '{config_path}'")
        config_module = importlibutil.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module
    config = load_config_module(configPath)

    # Export all attributes from the loaded config module
    for attr in dir(config):
        # print(attr)
        if not attr.startswith('_'):  # Skip private attributes
            globals()[attr] = getattr(config, attr)

    # let's just set up chats db as well while we're in here
    chatsPath = getChatsPath()
    if not os.path.exists(chatsPath):
        print("creating chats directory", chatsPath)
        os.makedirs(chatsPath, exist_ok=True)

setupConfig()