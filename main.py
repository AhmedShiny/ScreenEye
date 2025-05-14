import asyncio
import threading
import tkinter as tk
from tkinter import messagebox

from langchain_core.messages import HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import pyautogui
from PIL import Image
import io
import base64
import speech_recognition as sr
from gtts import gTTS
import tempfile
import playsound
import pyttsx3
import queue
from dotenv import load_dotenv
import os

load_dotenv()

# System prompt for accessibility-focused assistant
SYSTEM_PROMPT = (
    "You are an assistant that helps visually impaired or blind people understand what is on their computer screen. "
    "Whenever you receive a screenshot, describe its content in detail, including text, layout, and any important visual elements. "
    "Be concise, clear, and helpful. "
    "Ignore the gui window start recording and stop recording, as it is not relevant to the task."
    "Don't include any artsteriks or any other special characters in your response.JUST INCLUDE PLAIN TEXT."
)


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# Custom handler to stream LLM tokens as they're generated
class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

# Initialize LLM with streaming enabled
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    convert_system_message_to_human=True,
    model_kwargs={"system": SYSTEM_PROMPT}
)

# Helper to capture screenshot and return as bytes
def capture_screenshot():
    screenshot = pyautogui.screenshot()
    img_byte_arr = io.BytesIO()
    screenshot.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()

# Function to get voice input and convert to text
def get_voice_input():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Speak now...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language="en-US")
            print(f"You (voice): {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, could not understand audio. Please try again.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None
    except Exception as e:
        print(f"Microphone error: {e}")
        return None

# Function to play LLM response as voice
def speak_text(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        temp_path = fp.name
    try:
        tts.save(temp_path)
        playsound.playsound(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

# Global queue and speaker thread for TTS
tts_queue = queue.Queue()

def tts_worker():
    engine = pyttsx3.init()
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak_text_stream(text):
    tts_queue.put(text)

class VoiceRecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Recorder for LLM")
        self.is_listening = False
        self.recording_thread = None
        self.audio = None
        self.label = tk.Label(root, text="Click 'Start Recording' to begin.", font=("Arial", 14))
        self.label.pack(pady=10)
        self.start_btn = tk.Button(root, text="Start Recording", command=self.start_recording, width=20, font=("Arial", 12))
        self.start_btn.pack(pady=5)
        self.root.bind('<Control-r>', lambda event: self.start_recording())
        

    def start_recording(self):
        if self.is_listening:
            return
        self.is_listening = True
        self.label.config(text="Listening... Speak now.", fg="green")
        speak_text_stream("Recording started Please speak")
        self.start_btn.config(state=tk.DISABLED)
        self.recording_thread = threading.Thread(target=self.listen_voice)
        self.recording_thread.start()

    def listen_voice(self):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                self.audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
        except Exception as e:
            self.audio = None
            self.root.after(0, lambda: self.label.config(text=f"Microphone error: {e}", fg="red"))
            speak_text_stream(f"Microphone error. {e}")
        finally:
            self.is_listening = False
            # Directly process audio after listening
            self.root.after(0, self.process_audio if self.audio else self.no_audio_captured)

    def no_audio_captured(self):
        self.label.config(text="No audio captured.", fg="red")
        speak_text_stream("No audio captured.")
        self.start_btn.config(state=tk.NORMAL)

    def process_audio(self):
        self.start_btn.config(state=tk.NORMAL)
        self.label.config(text="Processing...", fg="blue")
        recognizer = sr.Recognizer()
        try:
            text = recognizer.recognize_google(self.audio)
            self.root.after(0, lambda: self.label.config(text=f"You said: {text}", fg="black"))
            img_bytes = capture_screenshot()
            img_b64 = base64.b64encode(img_bytes).decode()
            response = ""
            def update_label_and_speak(chunk_text):
                nonlocal response
                response += chunk_text
                self.root.after(0, lambda: self.label.config(text=f"AI: {response}", fg="blue"))
                speak_text_stream(chunk_text)
            async def handle_chunks():
                async for chunk in llm.astream([
                    HumanMessage(content=[{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}}])
                ]):
                    update_label_and_speak(chunk.content)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(handle_chunks(), loop)
                else:
                    loop.run_until_complete(handle_chunks())
            except RuntimeError:
                asyncio.run(handle_chunks())
        except sr.UnknownValueError:
            self.root.after(0, lambda: self.label.config(text="Sorry, could not understand audio.", fg="red"))
            speak_text_stream("Sorry, could not understand audio.")
        except sr.RequestError as e:
            self.root.after(0, lambda e=e: self.label.config(text=f"Could not request results; {e}", fg="red"))
            speak_text_stream(f"Could not request results. {e}")
        except Exception as e:
            self.root.after(0, lambda e=e: self.label.config(text=f"Error: {e}", fg="red"))
            speak_text_stream(f"Error occurred. {e}")
        finally:
            self.audio = None



def run_voice_gui():
    root = tk.Tk()
    VoiceRecorderGUI(root)
    root.mainloop()


run_voice_gui()

