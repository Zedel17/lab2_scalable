import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from personas import PERSONAS

# ==============================
# Model loading
# ==============================

print("Loading fine-tuned model from HuggingFace...")

# Get HF token from Space secrets (optional, for private repos)
hf_token = os.environ.get("HF_TOKEN")

model_id = "Zedel17/llama_1b_merged_float16"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

# Load model with 8-bit quantization for CPU efficiency
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto",
    token=hf_token
)

print("Model loaded successfully!")

# ==============================
# Chat function
# ==============================

def chat_fn(message, history, persona_name):
    # Get the system prompt for the selected persona
    system_prompt = PERSONAS[persona_name]

    # Simple prompt format (not rebuilding full conversation)
    prompt = (
        system_prompt
        + "\n\n"
        + f"User: {message}\n"
        + "Assistant:"
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (exclude prompt)
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Clean hallucinated follow-ups
    reply = reply.split("User:")[0].strip()
    reply = reply.split("Assistant:")[0].strip()

    return reply


# ==============================
# Gradio UI
# ==============================

with gr.Blocks() as demo:


    gr.Markdown(
        """
        # 💼 Customer Support AI Assistant

        Welcome! I'm here to help with **orders**, **billing**, **technical support**, and **product recommendations**.

        Select a support mode below and start chatting. I'm available 24/7 to assist you!
        """
    )

    gr.Markdown("---")

    # Persona selector
    persona_selector = gr.Radio(
        choices=list(PERSONAS.keys()),
        value=list(PERSONAS.keys())[0],
        label="🎭 Choose Assistant Type",
        info="Select the type of customer service assistant you'd like to talk to"
    )

    chatbot = gr.Chatbot(
        height=500,
        show_label=False,
        value=[
            {
                "role": "assistant",
                "content": "👋 Hello! I'm your Customer Support Assistant. How can I help you today?"
            }
        ]
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your message here...",
            scale=8,
            show_label=False
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    clear_btn = gr.Button("🧹 Clear conversation")

    # Quick action examples
    gr.Markdown("**💡 Quick Actions** - Click any example to get started:")
    gr.Examples(
        examples=[
            ["Track my order #12345"],
            ["What's my account balance?"],
            ["I need help with technical issues"],
            ["Show me upgrade options"]
        ],
        inputs=msg,
        label=""
    )

    # ==============================
    # Button logic
    # ==============================

    def respond(message, chat_history, persona_name):
        reply = chat_fn(message, chat_history, persona_name)
        # Gradio expects messages format with role/content
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        return "", chat_history

    def reset_chat():
        # Reset to welcome message
        return [
            {
                "role": "assistant",
                "content": "👋 Hello! I'm your Customer Support Assistant. How can I help you today?"
            }
        ]

    send_btn.click(respond, [msg, chatbot, persona_selector], [msg, chatbot])
    msg.submit(respond, [msg, chatbot, persona_selector], [msg, chatbot])
    clear_btn.click(reset_chat, None, chatbot, queue=False)

    gr.Markdown(
        """
        ---
        ### ℹ️ Technical details
        - **Base Model:** meta-llama/Llama-3.2-1B
        - **Fine-tuning:** PEFT/LoRA on FineTome-100k dataset
        - **Format:** Merged float16 model
        - **Inference:** Transformers with 8-bit quantization (CPU only)
        - **Training:** Supervised Fine-Tuning (SFT) with 100 steps
        - **Context window:** 2048 tokens
        """
    )

demo.launch()
