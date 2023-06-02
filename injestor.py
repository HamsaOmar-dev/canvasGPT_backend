import os
from env import load_dotenv
from canvas_api import canvas_api
from pprint import pprint
from langchain.document_loaders import UnstructuredFileLoader
import urllib.request
from langchain.text_splitter import CharacterTextSplitter
import uuid
import json
from sentence_transformers import SentenceTransformer
import pinecone

load_dotenv()

e_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
pinecone.init(api_key=os.getenv('PINECONE_KEY'), environment="us-west1-gcp-free")
index = pinecone.Index("hamsas-canvas")

def create_embeddings(texts):
    return e_model.encode(list(texts), convert_to_numpy=True).tolist()

# str -> List[{text:str, ref: str}]
def get_all_files(course_id):
    page = 1
    all_files = []
    while True:
        files = canvas_api.get(f"/courses/{course_id}/files?page={page}&per_page=100")
        if len(files) == 0: break
        all_files += files
        page += 1
    return all_files

# str, List[{text:str, ref: str}] -> se[upserts data to vector db]
def create_indexes(course_id, loaders):
    for loader in loaders:
        try:
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            embeddings = create_embeddings([t.page_content for t in texts])

            vectors = []
            for i in range(len(texts)):
                metadata = texts[i].metadata
                metadata["data"] = texts[i].page_content
                vectors.append((str(uuid.uuid4()), embeddings[i], metadata))

            upsert_response = index.upsert(
                vectors=vectors,
                namespace=f"Course{course_id}"
            )
            print(upsert_response)
        except Exception as e:
            print(e)
    print(f"indexes added for: {course_id}")

# str, str -> se[saves course info to file]
def save_course_to_file(course):
    id = str(course["id"])
    course_obj = {
        "id": course["id"],
        "collection": "Course" + id,
        "name": course["name"]
    }
    path  = f"courses/{id}.json"
    with open(path, 'w') as fp:
        json.dump(course_obj, fp)
    print(f"saved course:  {course['name']} to {path}")


def save_to_device(url, filename, output="tmp/"):
    if not os.path.exists(output):
        os.makedirs(output)

    output_file = output + filename
    urllib.request.urlretrieve(url, output_file)
    return output_file

def create_index_for_all_courses():
    courses = canvas_api.get("/courses")
    for course in [courses[2]]:
        try:
            files = get_all_files(str(course["id"]))
            if len(files) != 0:
                loaders = []
                for file in files:
                    print(f"downloading file: {file}")
                    saved_path = save_to_device(file["url"], file["filename"])
                    loader = UnstructuredFileLoader(saved_path)
                    loaders.append(loader)
                print(f"creating indexes for {str(course['id'])}")
                create_indexes(str(course["id"]), loaders)
                save_course_to_file(course)
        except Exception as e:
            print("============A course failed=========")
            print(course)
            print(e)
            print("============A course failed=========")


if __name__ == "__main__":
    # courses = canvas_api.get("/courses")
    # for i, c in enumerate(courses):
    #     print(f'{c["id"]}:{i}')
    create_index_for_all_courses()
