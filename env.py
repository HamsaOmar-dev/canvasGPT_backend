import os
from dotenv import load_dotenv

PATH_TO_ENV=".env"

def load_env_vars():
    if os.path.exists(PATH_TO_ENV):
        load_dotenv(PATH_TO_ENV)
    else:
        raise RuntimeError('.env file not found')
