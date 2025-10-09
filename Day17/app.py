from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def summarize_text(text, max_chunk_size=1000):
    # Break long text into chunks
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    summaries = []

    for chunk in chunks:
        prompt = f"Summarize the following text:\n\n{chunk}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=150)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary)

    # Optionally combine summaries into a final short summary
    combined = " ".join(summaries)
    final_prompt = f"Summarize this text concisely:\n\n{combined}"
    final_inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True)
    final_outputs = model.generate(**final_inputs, max_new_tokens=150)
    final_summary = tokenizer.decode(final_outputs[0], skip_special_tokens=True)

    return final_summary

def main():
    pdf_path = "example.pdf"
    text = extract_text_from_pdf(pdf_path)
    summary = summarize_text(text)
    
    print("📄 PDF Summary:\n")
    print(summary)
    
if __name__ == '__main__':
    main()