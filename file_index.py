import pymupdf
import torch
from chonkie import RecursiveChunker
from sentence_transformers import SentenceTransformer
import re
import unicodedata

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
chunker = RecursiveChunker(chunk_size=750, min_characters_per_chunk=100)
TOP_K = 5


def clean_block(text):
    text = unicodedata.normalize("NFKC", text)        # ligatures: ﬁ -> fi
    text = text.replace("\u00ad", "")                 # soft hyphens
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)   # rejoin words split at a line end
    return " ".join(text.split())                     # remaining newlines -> spaces

def pdf_r(file_name):
    pages = []
    with pymupdf.open(file_name) as doc:
        for page in doc:
            blocks = page.get_text("blocks", sort=True)
            paragraphs = [clean_block(b[4]) for b in blocks if b[6] == 0]
            text = "\n\n".join(p for p in paragraphs if p)
            if text:
                pages.append({"page": page.number + 1, "text": text})
    return pages


# Function for chunking the scanned text
# index_doc -> list of {page, chunk text}
def embed_doc(dict_list):
    chunked_list = [] # contains {page, chunked text}
    for d in dict_list:
        page = d["page"]
        text = d["text"]
        chunks = chunker(text)

        for chunk in chunks:
            
            chunked_list.append(({"page": page, "text": chunk.text}))

    txt = [c["text"] for c in chunked_list] # flat list of chunked text, only text, no page
    text_embeddings = embedder.encode_document(txt, normalize_embeddings=True, convert_to_tensor=True)
    return {"chunks": chunked_list, "embeddings": text_embeddings}


def retrieve_text(data, question, k = TOP_K):
    embedded = data["embeddings"]
    chunk_list = data["chunks"]

    qst_embed = embedder.encode_query(question, normalize_embeddings=True, convert_to_tensor=True)
    sim_score = embedder.similarity(qst_embed, embedded)
    scores, indices = torch.topk(sim_score, k)

    retrieved_data = []
    for sc,ind in zip(scores[0],indices[0]):
        page = chunk_list[int(ind)]["page"]
        txt = chunk_list[int(ind)]["text"]
        retrieved_data.append({"page": page, "text": txt, "score": float(sc)})
    return retrieved_data


def format_context(retrieved): # small formatter for retrieved text ,thanks Claude!
    return "\n\n".join(f"[Page {r['page']}] {r['text']}" for r in retrieved)



# for t in retrieve_text(embed_doc(pdf_r("bert.pdf")), "What is the model architecture?", 8):
#     print(f"[Page {t["page"]}] \n\n Content:\n{t["text"]}")
