from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("shayekh/openrag_llama2_7b_8x135m")
model = AutoModelForCausalLM.from_pretrained(
    "shayekh/openrag_llama2_7b_8x135m",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="bfloat16"
)
inputs = """### Instruction:
"You are a question answering agent. Given a context and a question, your task is to answer the question based on the context.
## Instruction:
A 202 pound slab of grewwacke covered in runes on its face and side indicated the Scandinavians came to Minnesota in what century?
[Retrieval]<paragraph>
Knowledge 1: Kensington Runestone
The Kensington Runestone is a 202 lb slab of greywacke covered in runes on its face and side. 
[SEP] 
Knowledge 2: Kensington, Minnesota
Kensington is a city in Douglas County, Minnesota, United States. The population was 292 at the 2010 census. The city is notable in Minnesota history for being the place wh
ere the famous, if questionable, Kensington Runestone was first displayed. The stone tablet may indicate that Scandinavians had come to Minnesota in the 14th century. It is
 now at a museum in nearby Alexandria, Minnesota.
</paragraph>"""
inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
pred = model.generate(**inputs, max_length=512, do_sample=False, num_return_sequences=1)
print(tokenizer.decode(pred[:, inputs.input_ids.shape[1]:].cpu()[0], skip_special_tokens=False))