# Facial Emotion Recognition Using CNN, RNN, and Hybrid Architectures

## Overview
This project investigates the effectiveness of different neural network architectures for **Facial Emotion Recognition (FER)** using the FER2013 dataset. The research spans traditional **Convolutional Neural Networks (CNNs)**, modern pre-trained architectures like **VGG19**, **MobileNetV1**, and **ResNet50**, and **hybrid models** that combine CNNs with **Recurrent Neural Networks (RNNs)**, specifically LSTM layers. The aim is to evaluate these models in terms of performance, computational efficiency, and generalization.

## Dataset
The project uses the **FER2013 dataset**, a widely recognized benchmark in FER. The dataset contains:
- Grayscale images resized to 48x48 pixels.
- Seven emotion labels: **Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral**.
- Data split into training (80%), validation (10%), and testing (10%) subsets.

To meet the input requirements of different architectures, single-channel images are converted to three-channel images for models like VGG19 and ResNet50.

## Features
- **Traditional CNNs**: Custom-designed models with varying complexities.
- **Pre-trained Architectures**: 
  - VGG19 (trainable and non-trainable versions).
  - MobileNetV1 and MobileNetV2 (optimized for lightweight applications).
  - ResNet50 (with layer freezing and unfreezing strategies).
- **Hybrid Architectures**:
  - CNN-LSTM: Combines spatial feature extraction (CNN) with temporal pattern recognition (LSTM).
  - ResNet50-LSTM: Enhances hybrid modeling with pre-trained ResNet50.
  - Attention mechanisms: Integrated at both CNN and LSTM layers for improved feature focus.
- **Evaluation Metrics**:
  - Accuracy, precision, recall, F1-score, and AUC.
  - Computational time tracking.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/facial-emotion-recognition.git
   cd facial-emotion-recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the FER2013 dataset:
   - [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
   - Place it in the `data/` directory.
4. Preprocess the data:
   ```bash
   python preprocess.py
   ```

## Usage
1. **Train Models**:
   - Example for VGG19:
     ```bash
     python train.py --model vgg19 --epochs 50 --batch-size 64
     ```
   - Example for ResNet50-LSTM:
     ```bash
     python train.py --model resnet50-lstm --epochs 30 --batch-size 32
     ```
2. **Evaluate Models**:
   ```bash
   python evaluate.py --model vgg19
   ```
3. **Visualize Results**:
   ```bash
   python visualize.py
   ```

## Results
### Key Findings:
- **VGG19 (Trainable)** achieved the highest accuracy of **66%**, highlighting the importance of fine-tuning.
- **MobileNetV1** matched VGG19 in accuracy but required significantly fewer computational resources.
- **ResNet50** improved from **25% to 61% accuracy** after iterative layer tuning.
- **Hybrid Models**:
  - CNN-LSTM models faced challenges with integration but showed promise in combining spatial and temporal features.
  - ResNet50-LSTM achieved balanced performance across classes, with further improvements using attention mechanisms.

### Sample Performance Metrics (Accuracy, F1-Score):
| Model          | Accuracy | F1-Score |
|----------------|----------|----------|
| VGG19          | 66%      | 0.65     |
| MobileNetV1    | 66%      | 0.65     |
| ResNet50       | 61%      | 0.60     |
| CNN-LSTM       | 26%      | 0.11     |
| ResNet50-LSTM  | 56%      | 0.56     |

## Applications
- **Healthcare**: Diagnose mental health conditions like depression and anxiety through emotion monitoring.
- **Human-Computer Interaction**: Enable emotion-aware systems for enhanced user experiences.
- **Customer Service**: Provide personalized responses based on detected emotions.
- **Social Robotics**: Improve human-robot interaction by understanding user emotions.

## Future Work
- Enhance **hybrid models** by addressing input-output shape mismatches.
- Incorporate advanced **attention mechanisms**.
- Experiment with **ensemble methods** for improved accuracy.
- Address class imbalance in the dataset.

## References
- Carrier, P.L., Courville, A. (2013). Challenges in representation learning: Facial expression recognition challenge.
- Ming, Y., Qian, H., & Guangyuan, L. (2022). CNN-LSTM facial expression recognition method fused with two-layer attention mechanism.
- Gupta, S., Kumar, P., & Tekchandani, R. K. (2023). Facial emotion recognition based on deep learning models.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

