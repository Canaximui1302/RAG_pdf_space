import pymupdf
from chonkie import RecursiveChunker
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Read and keep page number info
def pdf_r(file_name):
    pages = []
    with pymupdf.open(file_name) as doc:
        for page in doc:
            blocks = page.get_text("blocks", sort=True)
            paragraphs = [" ".join(b[4].split()) for b in blocks if b[6] == 0]
            text = "\n\n".join(p for p in paragraphs if p)
            if text:
                pages.append({"page": page.number + 1, "text": text})
    return pages


chunker = RecursiveChunker(chunk_size=600)

# Function for chunking the retrieved text
# dict is a list of dictionary
def index_doc(dict_list):
    index_list = []
    for d in dict_list:
        page = d["page"]
        text = d["text"]
        chunks = chunker(text)

        for chunk in chunks:
            
            index_list.append(({"page": page, "text": chunk.text}))
    return index_list

def embed_doc(chunks):
    txt = [c["text"] for c in chunks]

    text_embeddings = model.encode_document(txt, normalize_embeddings=True, convert_to_tensor=True)
    question = ["How are you today?"]
    qst_embeddings = model.encode_query(question, normalize_embeddings=True, convert_to_tensor=True)

for i in index_doc(pdf_r("test2.pdf")):
    print(i)

