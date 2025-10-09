from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def main():
    sentence = "Give me the report by tomorrow."
    prompt = f"Rewrite the following sentence politely:\n{sentence}"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    
    polite_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("Polite Sentence:", polite_sentence)

if __name__ == '__main__':
    main()