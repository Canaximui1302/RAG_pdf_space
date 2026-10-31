import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "occ-ai/OCC-RAG-1.7B"

print(torch.__version__, torch.cuda.is_available())

print(torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f"free {free/1e9:.1f} GB / total {total/1e9:.1f} GB")
print("allocated:", torch.cuda.memory_allocated()/1e9, "GB")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,   # use torch.float16 if your GPU doesn't support bf16
    device_map="cuda",            # fails loudly instead of silently offloading to CPU
)
model.eval()
print("device:", next(model.parameters()).device)

# A RAG-style prompt: short context plus a question
context = [
{
"text": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\Give three tips for staying healthy.1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night."
}
]

print(context[0]["text"])
question = "Give three tips for staying healthy."
messages = [
    {"role": "user", "content": f"Context: {context[0]["text"]} \n\n Question: {question}"},
]

text = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,        # no <think> block
)
inputs = tok([text], return_tensors="pt").to(model.device)

def run(max_new_tokens):
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return new_tokens, tok.decode(new_tokens, skip_special_tokens=True)

run(8)  # warm-up: the first call is slower (CUDA kernels, caches)

torch.cuda.synchronize()
t0 = time.perf_counter()
new_tokens, answer = run(20000)
torch.cuda.synchronize()
dt = time.perf_counter() - t0

print(answer)
print(f"{len(new_tokens)} new tokens in {dt:.2f}s -> {len(new_tokens)/dt:.1f} tokens/s")