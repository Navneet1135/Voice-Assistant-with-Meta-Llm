import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os
import speech_recognition as sr
import torch

model_name = "meta-llama/Llama-2-7b-chat-hf"
token = "YOUR_HF_TOKEN_HERE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = AutoModelForCausalLM.from_pretrained(model_name, token=token).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)


def chatbot_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True).to(device)

    outputs = model.generate(**inputs)

    bot_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if bot_reply.startswith(user_input):
        bot_reply = bot_reply[len(user_input):].strip()

    return bot_reply

def text_to_audio(response, voice="default"):
    tts = gTTS(text=response, lang="en", slow=False)
    audio_path = "response.mp3"
    tts.save(audio_path)
    return audio_path

def process_input(text_input, voice_input):
    
    if voice_input:
        recognizer = sr.Recognizer()
        with sr.AudioFile(voice_input) as source:
            audio = recognizer.record(source)
        try:
            
            text_input = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            text_input = "Sorry, I did not understand that."
        except sr.RequestError:
            text_input = "Sorry, there was an error with the speech service."


    response = chatbot_response(text_input)
    audio_file = text_to_audio(response, voice_input)
    return response, audio_file

with gr.Blocks() as voice_assistant:
    gr.Markdown("""
    # Voice Assistant

    You can type or speak your questions, and I will respond in both text and audio.
    """)

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Enter your message here:")
            voice_input = gr.Audio(type="filepath", label="Or record your voice:")
            submit_btn = gr.Button("Submit")

        with gr.Column():
            response_output = gr.Textbox(label="AI Response:",lines=15,max_lines=20,show_copy_button=True)
            audio_output = gr.Audio(label="Listen to Response:")

    submit_btn.click(
        fn=process_input,
        inputs=[text_input, voice_input],
        outputs=[response_output, audio_output]
    )

voice_assistant.launch(share = True)
