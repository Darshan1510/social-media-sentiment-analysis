# **Reddit Sentiment Analysis with Streamlit**

This project is a **Sentiment Analysis Web Application** built using **Streamlit**. The app enables users to analyze the sentiment of Reddit comments using three different machine learning models: **LSTM**, **GRU**, and **CNN**. Users can select a model to evaluate a comment's sentiment, and the app also provides insights into the performance of each model.

---

## **Project Features**
- Input a Reddit comment and analyze its sentiment (Positive, Neutral, Negative).
- Choose from three pre-trained models:
  - **LSTM Model**
  - **GRU Model**
  - **CNN Model**
- Visualize confidence scores for predictions.
- Compare model performances using key metrics like accuracy, F1 score, precision, and recall.

---

## **Model Architectures**
### **1. LSTM Model**
- **Architecture**: 
  - Long Short-Term Memory (LSTM) networks are a type of Recurrent Neural Network (RNN) designed to capture long-term dependencies in sequential data.
  - Includes memory cells, input gates, forget gates, and output gates.
- **Strengths**: 
  - Excels in understanding long-term dependencies.
  - Performs well in tasks requiring context retention across sequences.

### **2. GRU Model**
- **Architecture**:
  - Gated Recurrent Unit (GRU) networks are a simplified version of LSTM with fewer gates.
  - Combines the forget and input gates into a single "update gate."
- **Strengths**:
  - Faster training compared to LSTM.
  - Slightly fewer parameters, making it less prone to overfitting.
  - Retains strong sequential modeling capabilities.

### **3. CNN Model**
- **Architecture**:
  - Convolutional Neural Networks (CNNs) are typically used for image data but can effectively extract features from text data by applying convolutional filters over word embeddings.
- **Strengths**:
  - Fast training and inference.
  - Effective at detecting local patterns in data (e.g., n-grams).
- **Limitations**:
  - Less suited for capturing long-term sequential dependencies.

---

## **Performance Metrics**
Below is a comparison of the three models on the test dataset:

| **Metric**       | **LSTM Model** | **CNN Model** | **GRU Model** |
|-------------------|----------------|---------------|---------------|
| **Test Loss**     | 0.7650         | 1.7723        | 0.9325        |
| **Test Accuracy** | 83.45%         | 82.96%        | 83.70%        |
| **F1 Score**      | 83.49%         | 82.73%        | 83.75%        |
| **Precision**     | 83.56%         | 82.92%        | 83.82%        |
| **Recall**        | 83.45%         | 82.96%        | 83.70%        |

---

## **Key Observations**
1. **Accuracy**: 
   - The GRU model achieves the highest accuracy (83.70%), slightly outperforming LSTM and CNN.

2. **F1 Score**:
   - GRU also leads with an F1 score of 83.75%, indicating a better balance between precision and recall.

3. **Precision and Recall**:
   - GRU slightly outperforms LSTM in both precision and recall, while CNN trails behind.

4. **Loss**:
   - LSTM has the lowest test loss (0.7650), indicating better generalization, but GRU still performs comparably.

5. **Training Speed**:
   - CNN trains the fastest, followed by GRU, with LSTM being the slowest.

---

## **Why These Differences?**
1. **Sequential Nature of Data**:
   - GRU and LSTM, designed for sequential dependencies, outperform CNN in handling the contextual nature of text.

2. **Architecture Efficiency**:
   - GRU's simplified structure allows faster convergence and less overfitting compared to LSTM.
   - CNN's lack of sequential memory limits its ability to capture long-term dependencies.

3. **Overfitting vs. Generalization**:
   - LSTM and GRU handle long-term dependencies better, which is critical for nuanced sentiment analysis.
   - CNN is faster but may lose contextual nuances due to its focus on local patterns.

---

## **Conclusion**
- **Best Model**: The GRU model emerges as the best-performing model in terms of accuracy, F1 score, precision, and recall, making it an excellent choice for sentiment analysis tasks.
- **Why Choose LSTM**: LSTMs are still strong contenders due to their low test loss and similar metrics, making them suitable for tasks with complex dependencies.
- **Why Choose CNN**: CNNs are a great choice when faster training and inference are prioritized, despite slightly lower performance.

---

## **Future Work**
1. Experiment with hybrid architectures like:
   - **CNN-LSTM**: Combine CNN’s feature extraction with LSTM’s sequential memory.
   - **CNN-GRU**: Merge CNN’s efficiency with GRU’s simplicity.
2. Explore transformer-based architectures (e.g., **BERT**, **GPT**) for improved context understanding.
3. Apply regularization techniques to further enhance GRU’s generalization.

---

## **Getting Started**
### **Requirements**
1. Python 3.8+
2. Libraries:
   - TensorFlow
   - Streamlit
   - Pickle

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/reddit-sentiment-analysis.git
   cd reddit-sentiment-analysis
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## **Project Structure**
```
.
├── app.py               # Streamlit application script
├── sentiment_analysis_cnn.h5         # Pre-trained CNN model
├── sentiment_analysis_gru.h5         # Pre-trained GRU model
├── sentiment_analysis_lstm.h5        # Pre-trained LSTM model
├── tokenizer.pkl         # Tokenizer file
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
├── .gitignore           # Git ignore file
```

---

## **Contributions**
Contributions, issues, and feature requests are welcome! Feel free to submit a pull request or open an issue.

---

## **License**
This project is licensed under the [MIT License](LICENSE).

---