import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

MODEL_DIR = "./rm_qwen-coder3B_final"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
# model.half()
model.eval()

@app.route("/score", methods=["POST"])
def score():
    """Endpoint to score a batch of texts with the reward model."""
    data = request.get_json(force=True)
    texts = data.get("texts", [])
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Request must contain a 'texts' list"}), 400

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    if logits.shape[1] == 1:
        scores = logits.squeeze(-1)
    else:
        scores, _ = torch.max(logits, dim=1)

    scores_list = scores.tolist()
    return jsonify({"scores": scores_list})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
