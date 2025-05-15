# 🚢 Titanic Survival Prediction

Proyek ini bertujuan untuk memprediksi kemungkinan kelangsungan hidup penumpang Titanic berdasarkan data dari [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic). Proyek ini disusun oleh:

**Kelompok R**  
**Nama:** Zain Afif  
**NIM:** 24293032  

---

## 📦 Dataset

Data diambil dari Kaggle:
- `train.csv` — data latih dengan label `Survived`
- `test.csv` — data uji tanpa label
- `gender_submission.csv` — contoh format pengumpulan

---

## 🧠 Pendekatan dan Model

Model yang digunakan: **Random Forest Classifier**, karena memberikan akurasi lebih baik dibanding Decision Tree dasar.  
Tutorial rujukan: [Alexis Cook's Titanic Tutorial](https://www.kaggle.com/code/alexisbcook/titanic-tutorial)

Fitur yang digunakan:
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

---

## 💻 Kode Lengkap

```python
# Titanic Survival Prediction - Kelompok R
# Nama: Zain Afif | NIM: 24293032

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Preprocessing
# Pilih fitur
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_data[features]
y = train_data['Survived']

# Gabungkan data train dan test untuk preprocessing seragam
combined = pd.concat([X, test_data[features]])

# Encoding kategori
combined['Sex'] = combined['Sex'].map({'male': 0, 'female': 1})
combined['Embarked'] = combined['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Tangani nilai yang hilang
imputer = SimpleImputer(strategy='median')
combined_imputed = pd.DataFrame(imputer.fit_transform(combined), columns=features)

# Bagi kembali ke data latih dan uji
X = combined_imputed.iloc[:len(train_data)]
X_test = combined_imputed.iloc[len(train_data):]

# Split data latih menjadi latih dan validasi
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
print(f"Akurasi model: {accuracy:.4f}")

# Prediksi untuk data uji
test_predictions = model.predict(X_test)

# Simpan ke file submission
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions})
output.to_csv('submission.csv', index=False)
print("File submission.csv telah dibuat.")
