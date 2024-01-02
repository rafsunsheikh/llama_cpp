from llama_cpp import Llama


# Global Variables
my_model_path = "llama.cpp/models/quantized_q4_1.gguf"
CONTEXT_SIZE = 512

# Load the model
zephyr_model = Llama(model_path = my_model_path, n_ctx = CONTEXT_SIZE)

# Define the Parameters
def generate_text_from_prompt(user_prompt, max_tokens = 100, temperature = 0.3, top_p = 0.1, echo = True, stop = ["Q", "\n"]):
    # Define the parameters
    model_output = zephyr_model(user_prompt, max_tokens = max_tokens, temperature = temperature, top_p = top_p, echo = echo, stop = stop)
    final_result = model_output["choices"][0]["text"].strip()
    return final_result

if __name__ == "__main__":
    my_prompt = "Once upon a time, there was a little girl "
    zephyr_model_response = generate_text_from_prompt(my_prompt)
    print(zephyr_model_response)