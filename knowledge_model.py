
############################ Noravshan bilim modellari ############################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ma'lumotlar yaratish
# Noravshan bilim modellarida, bilimlar oldindan aniqlangan emas, balki modelni o'qitish jarayonida o'rganiladi.
X = torch.randn(1000, 10)  # Kirish xususiyatlari (1000 ta namuna, 10 ta xususiyat)
y = torch.randint(0, 2, (1000, 1), dtype=torch.float32)  # Maqsad belgisi (0 yoki 1)

# Ma'lumotlarni DataLoader orqali yuklaymiz
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Sun'iy neyron tarmog'ini qurish
# Noravshan bilim modellarida tarmoq parametrlari (og'irliklar va ofsetlar) modelni o'qitish jarayonida o'rganiladi.
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 64)  # Birinchi qatlam
        self.fc2 = nn.Linear(64, 32)  # Ikkinchi qatlam
        self.fc3 = nn.Linear(32, 1)   # Chiqarish qatlam
        self.dropout = nn.Dropout(0.5)  # Overfittingni oldini olish uchun Dropout
        self.relu = nn.ReLU()  # ReLU aktivatsiya funksiyasi
        self.sigmoid = nn.Sigmoid()  # Sigmoid aktivatsiya funksiyasi (binariya tasniflash uchun)

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Birinchi qatlam orqali o'tish
        x = self.dropout(x)  # Dropout qo'llash
        x = self.relu(self.fc2(x))  # Ikkinchi qatlam orqali o'tish
        x = self.sigmoid(self.fc3(x))  # Chiqarish qatlam orqali o'tish
        return x

# Model yaratish
model = SimpleNN()

# Yo'qotish funksiyasi va optimizatorni aniqlash
# Noravshan bilim modellarida optimallashtirishning klassik usullari qo'llaniladi.
criterion = nn.BCELoss()  # Binariya krossentropiya yo'qotish funksiyasi
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizatori

# Modelni o'qitish
# Noravshan bilim modellarida o'qitish ma'lumotlar to'plami asosida amalga oshiriladi.
for epoch in range(10):  # 10 ta epoxa davomida o'qitamiz
    for batch_X, batch_y in dataloader:
        # Oldindan hisoblangan gradientlarni tozalash
        optimizer.zero_grad()
        
        # Oldingi qadam
        outputs = model(batch_X)
        
        # Yo'qotishni hisoblash
        loss = criterion(outputs, batch_y)
        
        # Orqaga tarqalish (backpropagation) va optimallashtirish
        loss.backward()
        optimizer.step()
    
    print(f'Epoxa {epoch+1}, Yo\'qotish: {loss.item():.4f}')

# Modelning aniqligini tekshirish
# Noravshan bilim modellari odatda yangi ma'lumotlar ustida yaxshi ishlash uchun o'qitiladi.
with torch.no_grad():
    outputs = model(X)
    predictions = (outputs > 0.5).float()
    accuracy = (predictions == y).float().mean()
    print(f'Model aniqligi: {accuracy * 100:.2f}%')
