import os
import json
from typing import List

import pinecone
import openai
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
import tiktoken

from env import load_dotenv

load_dotenv()

class Message(BaseModel):
    role: str
    content: str
class Messages(BaseModel):
    history: List[Message]
    message: str

app = FastAPI()

origins = [
    "https://canvas-gpt-2wltar9k0-st2-ev.vercel.app",
    "http://127.0.0.1:5173",
    "https://canvas-gpt-f-delta.vercel.app",
    "https://coursemind-dev.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

e_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
pinecone.init(api_key=os.getenv('PINECONE_KEY'), environment="us-west1-gcp-free")
index = pinecone.Index("hamsas-canvas")
openai.api_key = os.getenv('OPENAI_KEY')

def create_embeddings(texts):
    return e_model.encode(list(texts), convert_to_numpy=True).tolist()

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

# HELPERS

def read_json_files(folder_path):
    json_list = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                try:
                    json_data = json.load(file)
                    json_list.append(json_data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON file: {file_path}")

    return json_list

# ROUTES

@app.get("/")
async def root():
    return {"message": "backend is up!!"}

@app.get("/courses")
async def list_all_courses():
    return read_json_files("courses")


def generate_prompt(context, question):
    return f"""context: {context}
question: {question}

based on the context answer the question, if not mention that the context is irrelevant
"""

def conditional_insert(messages, history):
    if len(history) != 0:
        grouped = [history[-1]] + messages
        if num_tokens_from_messages(grouped) < 4096:
            return grouped
    return messages

@app.post("/chat/{course_id}")
async def chat_with_course(course_id: str, messages: Messages):
    query_response = index.query(
        namespace=f"Course{course_id}",
        top_k=1,
        include_values=True,
        include_metadata=True,
        vector=create_embeddings([messages.message])[0],
    )
    prompt = generate_prompt(query_response['matches'][0]['metadata']['data'], messages.message)

    print(prompt)

    new_messages = [{"role": "user", "content": prompt}]

    new_messages = conditional_insert(new_messages, [dict(mess) for mess in messages.history])

    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=new_messages)
    return {
        "history": messages.history.append(messages.message),
        "message": {"role": "assistant", "content": chat_completion.choices[0].message.content},
        "src": query_response['matches'][0]['metadata']['source'].split("/")[1]
    }
