import pymupdf
import gradio as gr

def gradio_fct(file_path):
    if file_path is None:
        return "No file uploaded"
    return f"Got: {file_path}"

with gr.Blocks() as demo:
    gr.Markdown(
    """
    <div style="text-align: center;">
        <h1 style="font-size: 3rem;">Voice News Assistant</h1>
        <h3>Record a request such as <em>"latest news about AI"</em>.</h3>
        <ol style="display: inline-block; text-align: left; font-size: 0.95rem; opacity: 0.75;">
            <li>Press <b>Record</b> and speak your request.</li>
            <li>Press <b>Stop</b> when you're done.</li>
            <li>Press <b>Summarize</b> and wait for the response.</li>
        </ol>
    </div>
    """
    )

    with gr.Row():
        gr.Column(scale=2)                      # empty left margin
        with gr.Column(scale=3, min_width=400):
            btn = gr.UploadButton("Your pdf file", variant="primary")
            summary_box = gr.Textbox(label="Summary", lines=5, elem_id="summary_box")
        gr.Column(scale=2)                      # empty right margin

    btn.click(gradio_fct, inputs=[btn], outputs=[summary_box])

theme = gr.Theme.from_hub("harsh8001/skymist")

theme = theme.set(
    block_title_text_color=theme.block_label_text_color,
    block_title_text_color_dark=theme.block_label_text_color_dark,
    block_title_text_size=theme.block_label_text_size,
    block_title_text_weight=theme.block_label_text_weight,
)

css = """
#summary_box {
    padding-top: 8px !important;
}

"""

demo.launch(theme=theme, css=css)