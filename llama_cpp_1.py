from llama_cpp import Llama

# Instanciate the model
my_aweseome_llama_model = Llama(model_path="llama.cpp/models/quantized_q4_1.gguf")

prompt = "Once upon a time, "
max_tokens = 500
temperature = 0.3
top_p = 0.1
echo = True
stop = ["Q", "\n"]


# Define the parameters
model_output = my_aweseome_llama_model(
       prompt,
       max_tokens=max_tokens,
       temperature=temperature,
       top_p=top_p,
       echo=echo,
       stop=stop,
   )
final_result = model_output["choices"][0]["text"].strip()

print(final_result)