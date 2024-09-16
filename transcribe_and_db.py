import yt_dlp
import whisper
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import DeepLake

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ACTIVELOOP_TOKEN = os.getenv('ACTIVELOOP_TOKEN')

my_activeloop_org_id = "dash"
dataset_name = "youtube_dyson"

def download_mp4_from_youtube(urls, job_id):

    video_info = []

    for i, url in enumerate(urls):
        # Options
        file_temp = f'./{job_id}_{i}.mp4'
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]', # (or)'mp4'
            'outtmpl': file_temp,
            'quiet': True,
        }

        # Downloading the video file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            title = result.get('title', "")
            author = result.get('uploader', "")

        video_info.append((file_temp, title, author))

    return video_info

urls=["https://www.youtube.com/watch?v=3YgcoCbVMrI", # 8:40
    "https://www.youtube.com/watch?v=SF2OPcHN3YU", # 12 min; vacuums
      "https://www.youtube.com/watch?v=x9Gx2YT6tSg", # 13:31; vs
      "https://www.youtube.com/watch?v=dS0oFmzU06g"] # 6 min; fan
vides_details = download_mp4_from_youtube(urls, 1)

# --- << transcribing >>

model = whisper.load_model("base")

results = []
for video in vides_details:
    result = model.transcribe(video[0])
    results.append( result['text'] )
    print(f"Transcription for {video[0]}:\n{result['text']}\n") # debug

with open('text.txt', 'w') as file:
    file.write("\n\n".join(results))

with open('text.txt') as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", " ", ",", "\n"] 
)

texts = text_splitter.split_text(text)
docs = [Document(page_content=t) for t in texts]

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
dataset_path = f"hub://{my_activeloop_org_id}/{dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
db.add_documents(docs)
