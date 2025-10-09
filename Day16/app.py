from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def run_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def summarize_text(text):
    prompt = f"Summarize the following text in 3 concise sentences:\n\n{text}"
    return run_model(prompt)

def extract_keywords(summary):
    prompt = f"Extract 5 important keywords from this summary:\n\n{summary}\n\nKeywords:"
    return run_model(prompt)

def summarize_and_extract_keywords(text):
    summary = summarize_text(text)
    keywords = extract_keywords(summary)
    return summary, keywords

def main():
    text = """
    Artificial Intelligence (AI) is transforming industries by enabling machines to learn from data, 
    recognize patterns, and make decisions with minimal human intervention. 
    From healthcare to finance, AI technologies are automating tasks, improving efficiency, 
    and uncovering insights that were previously impossible to detect.
    """

    summary, keywords = summarize_and_extract_keywords(text)

    print("Summary:\n", summary)
    print("\nKeywords:\n", keywords)

if __name__ == '__main__':
    main()