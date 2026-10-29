import pymupdf
import gradio as gr

def handle(file_path):
    if file_path is None:
        return "No file uploaded"
    return f"Got: {file_path}"

with gr.Blocks() as demo:
    pdf = gr.File(label="Upload a PDF", file_types=[".pdf"], file_count="single")
    status = gr.Textbox(label="Status")
    pdf.upload(handle, inputs=pdf, outputs=status)

demo.launch()