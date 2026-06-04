import pyttsx3


def main():
    print("Inicjalizacja Modułu Mowy (TTS)...")

    engine = pyttsx3.init()

    rate = engine.getProperty('rate')
    engine.setProperty('rate', 160)

    voices = engine.getProperty('voices')

    engine.setProperty('voice', voices[1].id)

    tekst = "Hello. All systems are online."

    print(f"Mówię: '{tekst}'")

    engine.say(tekst)
    engine.runAndWait()

    print("Test Ukończony.")


if __name__ == "__main__":
    main()