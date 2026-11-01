import pymupdf
import gradio as gr
import file_index as fi
import llm

def get_embed_content(file_path):
    if file_path is None:
        return "No file uploaded"

    file_content = fi.pdf_r(file_path)
    if file_content == None:
        return "404 not found"
    embedded_content = fi.embed_doc(file_content)
    return embedded_content, f"Ready: {len(embedded_content['chunks'])} chunks from {len(file_content)} pages. Ask your question below."


def chatting(question, history, doc):
    if doc is None:
        return "Please upload a PDF first."
    retrieved_context = fi.retrieve_text(doc, question, 6)
    retrieved_context = fi.format_context(retrieved_context)

    _, answer = llm.send_request(retrieved_context, question)

    return answer
    

with gr.Blocks() as demo:
    doc_state = gr.State(None)
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
            btn = gr.UploadButton("Your pdf file", file_types=[".pdf"], variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
        gr.Column(scale=2)                      # empty right margin

    btn.upload(get_embed_content, inputs=[btn], outputs=[doc_state, status])
    gr.ChatInterface(chatting, additional_inputs=[doc_state])

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