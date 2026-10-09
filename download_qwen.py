from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-0.6B",
    local_dir="models/qwen3-0.6b"
)

print("Model downloaded to:", model_dir)