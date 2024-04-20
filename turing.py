 class TuringTest:
    def __init__(self):
        self.questions = [
            "Sizning eng sevimli kitobingiz nima?",
            "Oxirgi sayohatingizda qayerga borgansiz?",
            "Sizni nima qiziqtiryapti?",
            "Eng yaqin do'stingiz kim?",
            "Dunyoning qayerida yashashni xohlaysiz?"
        ]
        self.responses = [
            "Mening eng sevimli kitobim '1984'.",
            "Men oxirgi safarimda Osiyo bo'ylab sayohat qildim."
            "Men kompyuterlar bilan moslashishni o'rganyapman."
            "Men eng yaqin do'stimga tashrif buyurishni yoqtiraman."
            "Men Himoloyda yashashni xohlardim."

        ]

    def simulate_conversation(self):
        conversation = ""
        for i in range(5):
            question = random.choice(self.questions)
            response = random.choice(self.responses)
            conversation += "Odam: " + question + "\n"
            conversation += "Kompyuter: " + response + "\n"
        return conversation

def main():
    turing_test = TuringTest()
    conversation = turing_test.simulate_conversation()
    print("Turing testiga xush kelibsiz!\n")
    print("Mana, odam va mashina o'rtasidagi suhbat:\n")
    print(conversation)
    print("\nMashina gapirganmi yoki odammi?")

    judge_input = input("Agar siz uni mashina deb hisoblasangiz 'kompyuter' so'zini, agar siz uni odam deb hisoblasangiz “inson” so'zini kiriting.: ")

    if judge_input.lower() == "kompyuter":
        print("\nSiz bu mashina ekanligini taxmin qildingiz. To'g'ri!")
    elif judge_input.lower() == "odam":
        print("\nSiz bu odam ekanligini taxmin qildingiz. Noto'g'ri!")
    else:
        print("\nYaroqsiz kiritish. Iltimos, 'kompyuter' yoki 'odam' so'zlarini kiriting.")

if __name__ == "__main__":
    main()
