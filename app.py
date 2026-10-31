"""
app.py — Hugging Face Spaces Entrypoint for LeafMind
Mounts the LeafMind FastAPI backend onto Gradio so it runs on free managed Python CPU hardware.
"""
import gradio as gr
from src.api.app import app as fastapi_app

with gr.Blocks(title="LeafMind API Server") as demo:
    gr.Markdown("# 🍃 LeafMind Multi-Agent Crop Advisory API")
    gr.Markdown(
        "The LeafMind backend is active and running.\n\n"
        "- **Interactive Swagger Docs**: [Open /docs](/docs)\n"
        "- **System Health Probe**: [Open /health](/health)\n"
        "- **Metadata Endpoint**: [Open /meta](/meta)\n\n"
        "Frontend clients (e.g. Vercel) can send requests to `/analyze`, `/case/{case_id}`, etc."
    )

# Mount the Gradio demo onto our FastAPI app so all FastAPI routes (/analyze, /health, /meta, /docs) are exposed!
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

