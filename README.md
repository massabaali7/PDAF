# PDAF: Phonetic Debiasing Attention Framework for Speaker Verification
Paper: https://arxiv.org/pdf/2409.05799

Colab: https://colab.research.google.com/drive/11Pve_TOxcHxK1kQBZHRGdcpCim-EVH_7?usp=sharing

Speaker verification systems are crucial for authenticating identity
through voice. Traditionally, these systems focus on comparing feature vectors, overlooking the speech’s content. However, this paper challenges this by highlighting the importance of phonetic dominance, a measure of the frequency or duration of phonemes, as a
crucial cue in speaker verification. A novel Phoneme-Debiasing Attention Framework (PDAF) is introduced, integrating with existing
attention frameworks to mitigate biases caused by phonetic dominance. PDAF adjusts the weighting for each phoneme and influences
feature extraction, allowing for a more nuanced analysis of speech.
This approach paves the way for more accurate and reliable identity
authentication through voice. Furthermore, by employing various
weighting strategies, we evaluate the influence of phonetic features
on the efficacy of the speaker verification system.

## Features
- **Phoneme alignment for audio**: Aligns audio files to extract phoneme-specific information.
- **Transformer-based model for speaker embedding**: Uses a self-attention model to generate speaker embeddings.
- **Cosine similarity for verification**: Computes similarity between embeddings to verify if two audio samples belong to the same speaker.

## Requirements
To run this code, you'll need to install the following dependencies:
- `pip install torch torchvision torchaudio`
- `pip install datasets transformers`
- `pip install g2p_en praatio librosa g2pM`

Ensure you have the required custom modules:
- `chairsu_align`: for phoneme alignment.
- `ast_processor`: for audio feature extraction.
- `util_stats`: for extracting phoneme probabilities.
- `encoder.self_attn`: for the self-attention transformer model.

## Usage
### Configuration
To use this script, ensure you have a valid `config.yaml` file in the root directory. This file should include the following keys:

- `inference`: Configuration for inference, including model hyperparameters like `d_model`, `heads`, `numSpks`, `emb_dim`, `lambda_value`, and `threshold`.
- `data`: Configuration for data processing, including `sample_rate` and `checkpoint_path` for model weights.

### Running the Code
This script provides two main functions:

1. **`attn_model(audio)`**: Generates a speaker embedding for a given audio file.
2. **`verify(audio1, audio2)`**: Compares two audio embeddings and prints whether they match, based on a similarity threshold.

To run the code:

```python
wav_file1 = "./samples/00003_female.wav"
y_pred_1 = attn_model(wav_file1)

wav_file2 = "./samples/00005_female.wav"
y_pred_2 = attn_model(wav_file2)

score = verify(y_pred_1, y_pred_2)
print(score)
```

The function `verify()` takes the speaker embeddings generated from `attn_model()` and computes their similarity score using cosine similarity. The threshold for matching is specified in the configuration file.

### Explanation of Functions
- **`load_config(config_path)`**: Loads the configuration file in YAML format.
- **`preprocess_attn_model(wav_path, target_sample_rate, emb_dim)`**: Extracts audio features, performs resampling, aligns phonemes, and returns input vectors needed for the model.
- **`attn_model(audio)`**: Loads a self-attention transformer model from the checkpoint and generates an embedding for a given audio sample.
- **`verify(audio1, audio2)`**: Uses cosine similarity to compare two speaker embeddings, determining whether they belong to the same speaker.

## Directory Structure
The expected project directory structure:

```
.
├── samples
│   ├── 00003_female.wav
│   └── 00005_female.wav
├── config.yaml
├── main.py (the script provided above)
└── README.md
```

## Notes
- Make sure to update the paths to your own checkpoints and audio files.
- The threshold for verification can be adjusted based on desired sensitivity, using the `config.yaml` file.

## Contributing
Feel free to submit pull requests or open issues for any improvements or bug fixes.
