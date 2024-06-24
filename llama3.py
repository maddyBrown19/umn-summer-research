import ollama
# Need to code all three stages with Llama
my_model = "llama3"
my_prompt = [{"role": "user", "content": "How should Amphetamine be used and what is the dosage?"}]
response = ollama.chat(model = my_model, messages = my_prompt)
print(response["message"]["content"])
