import time
import json
from transformers import AutoTokenizer, AutoModel
import torch

NUM_SENTENCES = 50
OUTPUT_FILE = "model_performance.json"

def main():
    sentences = [f"This is sentence number {i}" for i in range(NUM_SENTENCES)]

    models = {
        "BERT": "bert-base-uncased",
        "DistilBERT": "distilbert-base-uncased"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for model_name, model_path in models.items():
        print(f"Loading {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)
        model.eval()

        print(f"Running inference for {model_name}...")

        total_time = 0.0
        with torch.no_grad():
            for sentence in sentences:
                inputs = tokenizer(sentence, return_tensors="pt").to(device)
                
                start_time = time.perf_counter()
                _ = model(**inputs)
                end_time = time.perf_counter()
                
                total_time += (end_time - start_time)
        
        avg_time = total_time / NUM_SENTENCES

        results[model_name] = {
            "total_sentences": NUM_SENTENCES,
            "total_time_sec": round(total_time, 4),
            "avg_inference_time_sec": round(avg_time, 4),
            "device": str(device)
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Benchmark completed! Results saved to '{OUTPUT_FILE}'")
    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    main()