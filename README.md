# 🎵 Music Genre Classification

An advanced machine learning system for automatic music genre classification using the GTZAN dataset. This project achieves 87.1% accuracy in classifying music into 10 different genres using audio features and ensemble learning techniques.

## 📊 Performance Highlights

- **Overall Accuracy**: 87.1%
- **Best Performing Genre**: Metal (96% F1-score)
- **Number of Genres**: 10 (Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock)
- **Dataset Size**: 1000 audio tracks (100 per genre)

## 🎯 Key Features

- Advanced audio feature extraction using Librosa
- Custom preprocessing pipeline for audio data
- Ensemble learning with stacking classifier
- Comprehensive genre-specific performance metrics
- Support for 3-second and 30-second audio segments

## 🛠️ Technical Stack

- **Python Libraries**: 
  - Librosa (audio processing)
  - Scikit-learn (machine learning)
  - Pandas & NumPy (data manipulation)
  - Matplotlib & Seaborn (visualization)
- **Audio Features**:
  - MFCCs (Mel-frequency cepstral coefficients)
  - Spectral features (centroid, bandwidth, rolloff)
  - Temporal features (RMS energy, zero-crossing rate)
  - Mel spectrograms

## 📈 Model Performance

| Genre     | Precision | Recall | F1-Score |
|-----------|-----------|---------|-----------|
| Metal     | 0.96      | 0.95    | 0.96      |
| Classical | 0.93      | 0.93    | 0.93      |
| Blues     | 0.90      | 0.90    | 0.90      |
| Jazz      | 0.85      | 0.90    | 0.88      |
| Hip-Hop   | 0.89      | 0.85    | 0.87      |
| Rock      | 0.81      | 0.77    | 0.79      | 

## 🔧 Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/music-genre-classification.git
cd music-genre-classification
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the GTZAN dataset and place it in the `data` directory:
```bash
data/
├── features_3_sec.csv
├── features_30_sec.csv
└── audio/
    ├── blues/
    ├── classical/
    ├── country/
    └── ...
```

## 💻 Usage

1. Preprocess audio files:
```python
python preprocess.py --input_dir data/audio --output_dir data/processed
```

2. Train the model:
```python
python train.py --input_data data/features_3_sec.csv --model_output models/genre_classifier.pkl
```

3. Make predictions:
```python
python predict.py --input_file path/to/audio.wav --model models/genre_classifier.pkl
```

## 📋 Project Structure

```
.
├── data/                  # Dataset files
├── models/               # Saved model files
├── notebooks/           # Jupyter notebooks
├── src/                 # Source code
│   ├── preprocess.py   # Audio preprocessing
│   ├── features.py     # Feature extraction
│   ├── train.py        # Model training
│   └── predict.py      # Prediction script
├── requirements.txt    # Dependencies
└── README.md          # Project documentation
```

## 🔄 Future Improvements

1. **Genre Classification Site**:
   - Support for YouTube links
   - Spotify API integration
   - Real-time audio classification

2. **Song Merger Platform**:
   - Audio clustering capabilities
   - Song averaging functionality
   - Cross-genre mixing features

## 📚 Dataset

This project uses the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), which includes:
- 1000 audio tracks
- 10 genres (100 tracks each)
- 30-second duration per track
- 22050Hz Mono 16-bit audio files

## 📝 Citation

```bibtex
@misc{gtzan-dataset,
  title={GTZAN Dataset - Music Genre Classification},
  author={Tzanetakis, George and Cook, Perry},
  year={2002},
  publisher={Kaggle}
}
```

## ✨ Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 👥 Authors

- Francisco Figueroa
- Cole Aydelotte

## 🙏 Acknowledgments

- GTZAN Dataset creators
- Librosa development team
- Scikit-learn community
- Kaggle for hosting the dataset
