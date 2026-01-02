# Stock Price Prediction - Up/Down Evaluation

Ein Machine Learning System zur Vorhersage von Aktienkursbewegungen (Up/Down) mit 10 verschiedenen ML-Modellen.

---

## 📋 Projektstruktur

```
ESA_2/
├── up_down_evalutation_all_v1.4.py    
├── data/
│   └── fundamentals_v1
│   └── prices_v3
├── Klassifikationsreports/                             
├── Model_comparison/                           
├── results.json  
└── README.md                         
```

---

## 🚀 Installation & Setup

### **1. Python-Abhängigkeiten installieren**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install tensorflow keras
pip install lightgbm xgboost catboost
pip install pytorch-tabnet
```

---

## Skript starten

### **Einfache Nutzung (alles automatisch)**

```bash
python up_down_evalutation_all_v1.4.py
```

Das Skript:
1. Lädt die Daten
2. Erstellt Features & Target-Variable
3. Trainiert alle 10 Modelle
4. Evaluiert Performance
5. Speichert beste Modelle
6. Macht Vorhersagen basierend auf dem besten Modell

---

## 📊 Ausgabe

Nach dem erfolgreichen Run erhältst du:

### **Console Output (Beispiel):**
```
[1/10] Training LSTM...
   ✓ F1=0.718, AUC=0.526, Acc=0.562

[2/10] Training Transformer...
   ✓ F1=0.718, AUC=0.501, Acc=0.560

...

======================================================================
ERGEBNISSE:
======================================================================
                 Modell       F1      AUC      Acc
                 TabNet 0.718074 0.526171 0.561632
                   LSTM 0.717967 0.503566 0.560763
            Transformer 0.717538 0.500794 0.559500
                 1D-CNN 0.717538 0.500541 0.559500
                XGBoost 0.717331 0.516252 0.559248
               LightGBM 0.717215 0.504449 0.559108
               CatBoost 0.715615 0.506668 0.558407
Hybrid LSTM+Transformer 0.715363 0.501710 0.560342
               Stacking 0.715150 0.486559 0.557846
          Random Forest 0.701794 0.515582 0.557145
======================================================================

🏆 BESTES MODELL: TabNet (F1=0.718)
```

## 🎯 Modelle im Überblick

| # | Modell | Typ | Best für |
|---|--------|-----|----------|
| 1 | LSTM | Deep Learning | Zeitreihen mit langfristigem Gedächtnis |
| 2 | Transformer | Deep Learning | Parallele Verarbeitung, schneller |
| 3 | 1D-CNN | Deep Learning | Lokale Muster in Sequenzen |
| 4 | Hybrid LSTM+Transformer | Deep Learning | Kombination beider Stärken |
| 5 | TabNet | Gradient Boosting | Interpretierbar, strukturierte Daten |
| 6 | LightGBM | Gradient Boosting | Schnell & effizient |
| 7 | XGBoost | Gradient Boosting | Robust & beliebt |
| 8 | CatBoost | Gradient Boosting | Mit kategorialen Features |
| 9 | Random Forest | Tree Ensemble | Robust & interpretierbar |
| 10 | Stacking Ensemble | Meta-Learner | Kombiniert mehrere Modelle |

---


