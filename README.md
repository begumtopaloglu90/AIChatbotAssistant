# AIChatbotAssistant

AIChatbotAssistant is a project that runs a Gradio server to interact with a chatbot powered by a language model.

## Setup

1. **Install dependencies**  
   Before running the project, you need to install the required packages.  
   Run the following command to install them:
   ```bash
   pip install -r requirements.txt
Download the Model
The chatbot model will be downloaded automatically into a folder named model. Once downloaded, the model will be cached for future use.

Run the Gradio Server
After installing the dependencies and downloading the model, you can start the Gradio server by running:


```python
python llama_chatbot.py
```
## Access the Chatbot
Once the server is running, you can interact with the chatbot through a web interface. Follow the instructions provided in the terminal to access it.

## Notes
The model is cached after being downloaded, so subsequent runs will use the cached version instead of downloading it again.
Ensure that all required dependencies are installed before running the server.
