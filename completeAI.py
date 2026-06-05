import torch
import pyttsx3
import sounddevice as sd
import numpy as np
import cv2
from PIL import Image
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings

warnings.filterwarnings("ignore")


class CompleteAI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[SYSTEM] Inicjalizacja rdzenia CompleteAI na: {self.device}")

        print("[SYSTEM] Konfiguracja syntezatora mowy...")
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)

        print("[SYSTEM] Ładowanie modułu słuchu (Whisper-tiny)...")
        self.whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(self.device)

        print("[SYSTEM] Ładowanie modułu wzroku (BLIP-base)... To pobierze ok. 900 MB za pierwszym razem!")
        self.vision_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(
            self.device)

        print("\n" + "=" * 50)
        self.speak("All systems are online. I am ready.")
        print("=" * 50)

    def speak(self, text):
        """Moduł Mowy: Wypisuje tekst i czyta go na głos."""
        print(f"AI: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self, duration=4):
        """Moduł Słuchu: Nagrywa z mikrofonu i tłumaczy na tekst."""
        print(f"\nSłucham przez {duration} sekundy...")
        fs = 16000
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()

        print("Analiza fal dźwiękowych...")
        input_features = self.whisper_proc(audio.flatten(), sampling_rate=fs, return_tensors="pt").input_features.to(
            self.device)
        predicted_ids = self.whisper_model.generate(input_features, language="en")
        transcription = self.whisper_proc.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip().lower()

        print(f"Usłyszałem: '{transcription}'")
        return transcription

    def see_and_describe(self):
        """Moduł Wzroku: Odpala kamerę, robi zdjęcie i tłumaczy obraz na słowa."""
        print("📸 Uruchamianie kamery i robienie zdjęcia...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "error reading camera"

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        print("Przetwarzanie obrazu przez sieć neuronową...")
        inputs = self.vision_proc(image, return_tensors="pt").to(self.device)
        out = self.vision_model.generate(**inputs)
        description = self.vision_proc.decode(out[0], skip_special_tokens=True)

        return description

    def run(self):
        """Główna pętla asystenta."""
        while True:
            user_input = input("\nNaciśnij [ENTER], aby wydać polecenie głosowe (lub wpisz 'exit')...")
            if user_input.lower() == 'exit':
                self.speak("Shutting down the system. Goodbye.")
                break

            command = self.listen(duration=4)

            if not command:
                continue

            if "see" in command or "look" in command or "camera" in command:
                self.speak("Let me take a look.")
                description = self.see_and_describe()

                if description == "error reading camera":
                    self.speak("I'm sorry, I couldn't access your camera.")
                else:
                    self.speak(f"I think I see {description}.")

            elif "hello" in command or "hi" in command:
                self.speak("Hello there! How are you doing today?")

            elif "who are you" in command:
                self.speak("I am Complete A.I., your personal multimodal assistant.")

            elif "exit" in command or "stop" in command:
                self.speak("Going to sleep. Goodbye.")
                break

            else:
                self.speak(f"I heard you say: {command}, but I don't have a specific subroutine for that yet.")


if __name__ == "__main__":
    ai = CompleteAI()
    ai.run()