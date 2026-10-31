import pymupdf
import torch
from chonkie import RecursiveChunker
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
chunker = RecursiveChunker(chunk_size=600)

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


# Function for chunking the retrieved text
# index_doc -> list of {page, chunk text}
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
    # make a list of chunks := [c1, c2, ...]
    txt = [c["text"] for c in chunks]

    text_embeddings = model.encode_document(txt, normalize_embeddings=True, convert_to_tensor=True)

    question = ["what is transformer"]
    qst_embeddings = model.encode_query(question, normalize_embeddings=True, convert_to_tensor=True)

    sim_score = model.similarity(qst_embeddings, text_embeddings)

    scores, indices = torch.topk(sim_score, k=4)

    print(scores)
    print(indices[0])
    for sc,ind in zip(scores[0],indices[0]):
        print(f"[Score: {sc}]", chunks[ind]["text"])
    print("Next\n")




