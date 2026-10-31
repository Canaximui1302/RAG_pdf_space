import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import file_index as fi

MODEL = "Qwen/Qwen3-1.7B"
MAX_TOKEN = 250
# Define the rules for the model
SYSTEM = (
    "Answer the question using only the provided context. "
    "Reply in a natural, conversational way in 2 to 4 sentences, "
    "and mention the page numbers you used. "
    "If the context does not contain the answer, reply only: "
    "\"I couldn't find that in the document.\" Do not describe or guess what the context contains."
)

print(torch.__version__, torch.cuda.is_available())

print(torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f"free {free/1e9:.1f} GB / total {total/1e9:.1f} GB")
print("allocated:", torch.cuda.memory_allocated()/1e9, "GB")

tok = AutoTokenizer.from_pretrained(MODEL)
# use torch.float16 for faster per
model = AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16, device_map="cuda")  # fails loudly instead of silently offloading to CPU

model.eval()
print("device:", next(model.parameters()).device)


def send_request(context, question, max_token = MAX_TOKEN):
    messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": f"Context: {context} \n\n Question: {question}"},
    ]

    prompt = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,        # no <think> block
    )
    inputs = tok([prompt], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_token, do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return new_tokens, tok.decode(new_tokens, skip_special_tokens=True)


# Time  counter initialization
torch.cuda.synchronize()
t0 = time.perf_counter()
# new_tokens, answer = run(20000)

question = "How many parameters do GPT-2 have"
retrieved = fi.retrieve_text(fi.embed_doc(fi.pdf_r("bert.pdf")), question)
context = fi.format_context(retrieved)


new_tokens, answer = send_request(context, question)
print(answer)
print("generated tokens:", len(new_tokens))

# Time  counter end
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print("Executed in : ", dt, "s")