# Image Inpainting Pipeline: LaMa vs Stable Diffusion Comparison

**Praca magisterska**: "Wykorzystanie metod segmentacji oraz inpainting do korekty obrazów na podstawie opisu słownego"

Kompleksowy system do porównywania różnych metod inpaintingu obrazów z wykorzystaniem zaawansowanych technik detekcji obiektów, segmentacji i uzupełniania obrazów.

## Cel projektu

Projekt implementuje i porównuje dwie zaawansowane metody inpaintingu obrazów:
- **LaMa (Large Mask Inpainting)** - metoda oparta na architekurze Fast Fourier Convolution
- **Stable Diffusion Inpainting** - metoda wykorzystująca modele dyfuzyjne

System automatycznie wykrywa obiekty, tworzy maski segmentacji i usuwa wybrane elementy z obrazów, następnie ocenia jakość wyników przy użyciu standardowych metryk ewaluacji.

## Architektura systemu

### Pipeline przetwarzania:
1. **Detekcja obiektów** → Grounding DINO
2. **Segmentacja precyzyjna** → Segment Anything Model (SAM)
3. **Inpainting** → LaMa lub Stable Diffusion
4. **Ewaluacja** → Wielowymiarowa analiza jakości

### Komponenty główne:

#### Grounding DINO
- Model detekcji obiektów sterowany tekstem
- Lokalizuje obiekty na podstawie opisów w języku naturalnym
- Wersja: SwinT OGC

#### Segment Anything Model (SAM) 
- Precyzyjna segmentacja wykrytych obiektów
- Model: ViT-H (4.8B parametrów)
- Automatyczna dilatacja masek (15px) dla lepszego pokrycia

#### Metody Inpainting

**LaMa (Large Mask Inpainting)**
- Architektura: Fast Fourier Convolution
- Model: big-lama (pretrenowany)

**Stable Diffusion Inpainting**
- Model: stabilityai/stable-diffusion-2-inpainting
- Architektura: Latent Diffusion Model

## System ewaluacji

### Metryki jakości:

#### PSNR (Peak Signal-to-Noise Ratio)
- Zakres: 0-∞ dB (wyższe = lepsze)
- Mierzy różnicę piksel po pikselu

#### SSIM (Structural Similarity Index)
- Zakres: 0-1 (wyższe = lepsze) 
- Uwzględnia percepcję wzrokową człowieka
- Analizuje luminancję, kontrast i strukturę

#### LPIPS (Learned Perceptual Image Patch Similarity)
- Zakres: 0-∞ (niższe = lepsze)
- Wykorzystuje głębokie sieci neuronowe
- Najlepiej skorelowana z oceną ludzkiej percepcji

#### FID (Fréchet Inception Distance)
- Zakres: 0-∞ (niższe = lepsze)
- Mierzy podobieństwo rozkładów cech
- Ocena globalnej jakości generowania

## Instalacja i konfiguracja

### Środowisko:
Projekt został stworzony do pracy w **Google Colab** z dostępem do GPU (T4 lub lepszy).

### Automatyczna instalacja:
```bash
# Wykonaj wszystkie komórki instalacyjne w kolejności:
# 1. Grounding DINO + zależności
# 2. Segment Anything Model  
# 3. LaMa + wymagane biblioteki
# 4. Stable Diffusion pipeline
```

### Struktura katalogów:
```
/content/
├── GroundingDINO/           # Kod źródłowy Grounding DINO
├── lama/                    # Implementacja LaMa
├── weights/                 # Wagi modeli
│   ├── groundingdino_swint_ogc.pth
│   └── sam_vit_h_4b8939.pth
└── drive/MyDrive/RORD/      # Dataset RORD
    ├── person/
    ├── car/
    └── [inne kategorie]/
```

## Format datasetu RORD

### Struktura plików:
```
RORD/
├── person/
│   ├── person_yes1.jpg      # Obraz z obiektem
│   ├── person_no1.jpg       # Ground truth (bez obiektu)
│   ├── person_inpainted_lama1.jpg    # Wynik LaMa
│   └── person_inpainted_sd1.jpg      # Wynik Stable Diffusion
├── car/
│   ├── car_yes1.jpg
│   ├── car_no1.jpg
│   └── ...
└── [inne kategorie]/
```

### Konwencja nazewnictwa:
- `{prompt}_{yes/no}{id}.jpg` - pary treningowe
- `{prompt}_inpainted_{method}{id}.jpg` - wyniki inpaintingu

## Użytkowanie

### 1. Testowanie pojedynczego obrazu:
```python
result = test_single_image_standalone_fixed(
    '/path/to/image.jpg',
    'person',  # obiekt do usunięcia
    method='lama'  # lub 'stable_diffusion'
)
```

### 2. Ewaluacja pełnego datasetu:
```python
# LaMa
results_lama = evaluate_custom_dataset(method="lama")

# Stable Diffusion  
results_sd = evaluate_custom_dataset(method="sd", pipe=pipe)
```

### 3. Wizualizacja wyników:
```python
# Najlepsze/najgorsze wyniki według wybranej metryki
show_best_worst_results(results_lama, sort_by='psnr', num_best=3, num_worst=3)

# Próbki wyników
show_sample_results(num_samples=3, method="lama")
```

### 4. Analiza statystyczna:
```python
# Porównanie metod z testami statystycznymi
compare_methods_stats(results_lama, results_sd, 
                     method1_name="LaMa", 
                     method2_name="Stable-Diffusion")
```

## Funkcje analizy statystycznej

### Testy normalności:
- **Shapiro-Wilk test** - dla małych próbek
- **Anderson-Darling test** - alternatywa dla różnych rozmiarów

### Porównania między metodami:
- **Paired t-test** - dla rozkładów normalnych
- **Wilcoxon signed-rank** - dla rozkładów nienormalnych
- **Cohen's d** - wielkość efektu
- **Bootstrap confidence intervals** - przedziały ufności

### Analiza korelacji:
- **Pearson correlation** - korelacja liniowa
- **Spearman correlation** - korelacja monottoniczna między metrykami

## Struktura wyjściowa

### Pliki wynikowe:
```
/content/
├── evaluation_results_lama.json     # Szczegółowe wyniki LaMa
├── evaluation_results_sd.json       # Szczegółowe wyniki SD
├── standalone_results/              # Pojedyncze testy
└── temp_inpainting/                # Pliki tymczasowe
```

### Format JSON wyników:
```json
{
  "method": "lama",
  "success_count": 85,
  "total_count": 100,
  "psnr_scores": [23.4, 25.1, ...],
  "ssim_scores": [0.78, 0.82, ...],
  "lpips_scores": [0.23, 0.19, ...],
  "fid_score": 45.2,
  "per_folder_results": {...},
  "per_prompt_results": {...}
}