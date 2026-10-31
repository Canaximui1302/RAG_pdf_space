import pymupdf
import gradio as gr
import file_index as pr
import llm

def gradio_fct(file_path):
    if file_path is None:
        return "No file uploaded"

    read = pr.pdf_r(file_path)
    if read == None:
        return "404 not found"
    return read

with gr.Blocks() as demo:
    gr.Markdown(
    """
    <div style="text-align: center;">
        <h1 style="font-size: 3rem;">PDF Q&A Assistant</h1>
        <h3>Upload your file and ask question like <em>"What is this file about?"</em>.</h3>
        <ol style="display: inline-block; text-align: left; font-size: 0.95rem; opacity: 0.75;">
            <li>Press <b>Upload</b> and choose your .pdf file.</li>
            <li>Use <b>Chat</b> to discuss with LLM on your file.</li>
        </ol>
    </div>
    """
    )

    with gr.Row():
        gr.Column(scale=2)                      # empty left margin
        with gr.Column(scale=3, min_width=400):
            btn = gr.UploadButton("Your pdf file", variant="primary")
            out = gr.Textbox(label="Result")
        gr.Column(scale=2)                      # empty right margin

    btn.upload(gradio_fct, inputs=[btn], outputs=[out])

theme = gr.Theme.from_hub("harsh8001/skymist")

theme = theme.set(
    block_title_text_color=theme.block_label_text_color,
    block_title_text_color_dark=theme.block_label_text_color_dark,
    block_title_text_size=theme.block_label_text_size,
    block_title_text_weight=theme.block_label_text_weight,
)

# css = """
# #summary_box {
#     padding-top: 8px !important;
# }

# """

demo.launch(theme=theme)