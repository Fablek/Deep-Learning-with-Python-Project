# Plan Projektu: Deep Learning - Regresja (16 zbiorów danych)

## Faza 1: Inicjalizacja Środowiska i Struktury (Fundamenty)
* **Zadanie 1.1: Konfiguracja środowiska**
  * Utworzenie wirtualnego środowiska Pythona (venv).
  * Instalacja bazowych bibliotek: TensorFlow/Keras, scikit-learn, pandas, numpy, matplotlib, seaborn.
  * Wygenerowanie pliku z zależnościami (requirements.txt).
* **Zadanie 1.2: Architektura katalogów**
  * Założenie folderów: `data/` (na 16 plików ze zbiorami), `src/` (na logikę), `results/` (na zapisane wykresy i tabelę) oraz `notebooks/`.
  * Utworzenie pliku głównego `main.py`.

## Faza 2: Warstwa Danych (Data Loader i Eksploracja)
* **Zadanie 2.1: Implementacja Data Loadera**
  * Zbudowanie modułu skanującego `data/` w poszukiwaniu 16 zbiorów danych.
  * Rozdzielenie danych na macierz cech wejściowych (X) i wektor docelowy (y = zmienna R).

## Faza 3: Warstwa Preprocesingu (Zarządzanie Transformacjami)
* **Zadanie 3.1: Abstrakcja skalowania (Wzorzec Strategii)**
  * Zdefiniowanie interfejsu transformacji danych.
* **Zadanie 3.2: Implementacja transformatorów**
  * Utworzenie klas skalujących (np. StandardScaler).
  * Skalowanie cech wejściowych niezależnie od zmiennej docelowej R. Zapewnienie funkcji odwracającej skalowanie (inverse transform) dla R.

## Faza 4: Architektura Sieci Neuronowej (Model Factory)
* **Zadanie 4.1: Fabryka Modeli (Model Factory)**
  * Klasa produkująca nieskompilowane modele bazowe (MLP) dla problemu regresji (funkcja liniowa na wyjściu).
  * Wystawienie hiperparametrów na zewnątrz (ilość warstw, neurony, optimizer).

## Faza 5: Silnik Ewaluacji (K-Fold Cross-Validation z Ekstrakcją Danych do Wykresów)
* **Zadanie 5.1: Pętla Walidacyjna**
  * Implementacja podziału K-Fold z ochroną przed wyciekiem danych (fit transformatora tylko na zbiorze treningowym foldu).
* **Zadanie 5.2: Agregacja metryk i danych do wizualizacji**
  * Oprócz zbierania średniego błędu (MSE/MAE) ze wszystkich foldów, silnik musi "złapać" z wybranego, jednego foldu (np. fold nr 1):
    * Historię uczenia (Loss w każdej epoce dla zbioru treningowego i walidacyjnego).
    * Tablice z predykcjami (Predicted R) i odpowiadającymi im wartościami rzeczywistymi (True R) ze zbioru walidacyjnego.

## Faza 6: Wizualizacja i Raportowanie (Kluczowe Wymaganie)
* **Zadanie 6.1: Moduł Wizualizacji (Plotter)**
  * Utworzenie funkcji generującej krzywe uczenia (Learning Curves) - Loss vs Epochs.
  * Utworzenie funkcji generującej wykresy punktowe (Scatter Plots) - True R vs Predicted R.
* **Zadanie 6.2: Generowanie Raportu Końcowego (`main.py`)**
  * Orkiestracja całego potoku dla wszystkich 16 zbiorów.
  * Wygenerowanie 16 wykresów krzywych uczenia i zapisanie ich w folderze `results/`.
  * Wygenerowanie 16 wykresów True vs Predicted i zapisanie ich w folderze `results/`.
  * Zbudowanie i zapisanie pojedynczej tabeli (np. w formacie CSV lub jako ładnie sformatowany DataFrame) zawierającej błędy (MSE/MAE) dla każdego z 16 zbiorów danych.