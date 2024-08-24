
############################ Ekspert tizimlari ############################

class ExpertSystem:
    def __init__(self):
        # Tizim qoidalari, bu yerda har bir qoida shart va natijadan iborat
        self.rules = [
            {"symptom": "isitma", "disease": "gripp"},
            {"symptom": "yo'tal", "disease": "gripp"},
            {"symptom": "tomoq og'rig'i", "disease": "gripp"},
            {"symptom": "bosh og'rig'i", "disease": "migrain"},
            {"symptom": "ko'ngil aynishi", "disease": "ovqatdan zaharlanish"},
            {"symptom": "qorin og'rig'i", "disease": "ovqatdan zaharlanish"},
            {"symptom": "nafas olish qiyinligi", "disease": "astma"},
            {"symptom": "ko'z qichishishi", "disease": "allergiya"},
        ]
        self.diagnosis = {}  # Tashxis natijalarini saqlash uchun

    def diagnose(self, symptoms):
        # Tashxis qo'yish jarayoni
        for rule in self.rules:
            for symptom in symptoms:
                if symptom == rule["symptom"]:
                    if rule["disease"] in self.diagnosis:
                        self.diagnosis[rule["disease"]] += 1
                    else:
                        self.diagnosis[rule["disease"]] = 1

        # Natijani chiqarish
        if not self.diagnosis:
            print("Tashxis topilmadi. Iltimos, boshqa simptomlar kiriting.")
        else:
            probable_disease = max(self.diagnosis, key=self.diagnosis.get)
            print(f"Eng ehtimoliy tashxis: {probable_disease}")

# Foydalanuvchidan simptomlarni kiritishni so'raymiz
symptoms = input("Simptomlaringizni vergul bilan ajratib kiriting (masalan: isitma, yo'tal): ").split(", ")

# Ekspert tizimini yaratamiz va diagnostika jarayonini boshlaymiz
system = ExpertSystem()
system.diagnose(symptoms)
