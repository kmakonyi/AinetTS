import os
import warnings
import configparser

import appdirs

CONFIG_FILE = appdirs.user_config_dir('assist')
if os.path.exists(CONFIG_FILE):
    parser = configparser.ConfigParser()
    parser.read(CONFIG_FILE)
    config_data = parser['all']
else:
    config_data = dict()

CACHE_DIR = config_data.get('cache_dir', appdirs.user_cache_dir('assist'))
os.makedirs(CACHE_DIR, exist_ok=True)
