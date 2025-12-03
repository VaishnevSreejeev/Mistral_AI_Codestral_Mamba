import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Mamba Codestral Chat", page_icon="🐍")

st.title("🐍 Mamba Codestral 7B Chat")

@st.cache_resource
def load_model():
    repo_id = "gabriellarson/Mamba-Codestral-7B-v0.1-GGUF"
    filename = "Mamba-Codestral-7B-v0.1-Q4_0.gguf"
    
    with st.spinner(f"Loading model {filename}..."):
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        
        # n_gpu_layers=-1 attempts to offload all layers to GPU if available.
        # n_ctx=4096 sets the context window.
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, 
            n_ctx=2048,
            verbose=True
        )
    return llm

try:
    llm = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask for code..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Format prompt for instruction tuning if needed, or just raw.
        # Simple instruction format:
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        stream = llm(
            formatted_prompt,
            max_tokens=1024,
            stop=["</s>"],
            stream=True,
            echo=False
        )
        
        for output in stream:
            token = output['choices'][0]['text']
            full_response += token
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
