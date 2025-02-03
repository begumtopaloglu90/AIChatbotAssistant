import gradio as gr
from llama_cpp import Llama
# Project :Ai Assistant
# Define the inference parameters
inference_params = {
    "n_threads": 4,  # Increase the number of threads (based on M1 Pro's multi-core architecture)
    "n_predict": 5000,  # Limit the number of predicted tokens
    "top_k": 20,  # Reduce the top_k to consider fewer candidates
    "min_p": 0.05,
    "top_p": 0.85,  # Lower the top_p for faster but still varied output
    "temp": 0.7,  # Reduce temperature for less randomness and faster responses
    "repeat_penalty": 1.1,
    "input_prefix": "<|start_header_id|>user<|end_header_id|>\\n\\n",
    "input_suffix": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n",
    "antiprompt": [],
    "pre_prompt": "Sen bir yapay zeka asistanısın. Kullanıcı sana bir görev verecek. Amacın görevi olabildiğince sadık bir şekilde tamamlamak.",
    "pre_prompt_suffix": "<|eot_id|>",
    "pre_prompt_prefix": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n",
    "seed": -1,
    "tfs_z": 1,
    "typical_p": 1,
    "repeat_last_n": 64,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n_keep": 0,
    "logit_bias": {},
    "mirostat": 0,
    "mirostat_tau": 5,
    "mirostat_eta": 0.1,
    "memory_f16": True,
    "multiline_input": False,
    "penalize_nl": True
}

# make reference to downloaded model to reuse it.
# model_path = "modelFolder/models--ytu-ce-cosmos--Turkish-Llama-8b-Instruct-v0.1-GGUF/snapshots/ccd7b54d9f933541dbf58b91e6d0e9830ca74472/Turkish-Llama-8b-Instruct-v0.1.Q4_K.gguf" # Give model path in your pc
# llama = Llama(model_path=model_path, verbose=False)

# use this object to download the model 
llama = Llama.from_pretrained(
    repo_id="ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1-GGUF",
    filename="*Q4_K.gguf",
    verbose=False,
    cache_dir="model"
)
# Function to generate a response based on user input
def generate_response(user_input):
    # Construct the prompt
    prompt = f"{inference_params['pre_prompt_prefix']}{inference_params['pre_prompt']}\n\n{inference_params['input_prefix']}{user_input}{inference_params['input_suffix']}"
    # Generate the response
    response = llama(prompt, max_tokens=inference_params["n_predict"])
    return response['choices'][0]['text']

# Define Gradio interface
def gradio_interface(user_input):
    response = generate_response(user_input)
    return response

# Create the Gradio interface
iface = gr.Interface(fn=gradio_interface, inputs="text", outputs="text", live=True, title="AI Assistant", description="Sorularınızı yazın ve cevap alın.")

# Launch the interface
iface.launch(share=False)