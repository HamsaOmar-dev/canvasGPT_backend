import os
import requests

from env import load_dotenv

load_dotenv()

class CanvasAPI:

    def __init__(self, token, base_url):
        self.access_token = token
        self.base_url = base_url

    def get(self, path):
        url = f'{self.base_url}/api/v1{path}'
        response = requests.get(url, headers={'Authorization': f'Bearer {self.access_token}'})
        if response.status_code == 200:
            return response.json()
        elif response.status_code  == 403:
            return []
        else:
            raise RuntimeError(f'Error retrieving {url}. Status code: {response.status_code}. Failed because {str(response.content)}')

canvas_api = CanvasAPI(os.getenv('CANVAS_KEY'), 'https://canvas.umn.edu')