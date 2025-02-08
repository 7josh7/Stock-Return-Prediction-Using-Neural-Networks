# Stock Return Prediction Using Neural Networks

## Project Overview
This project applies machine learning techniques, specifically Neural Networks, to predict the next day's return of Apple Inc. (AAPL) stock. It demonstrates expertise in data preprocessing, feature engineering, and deep learning-based forecasting.

## Objectives
- **Stock Market Data Analysis:** Understanding historical stock trends and movements.
- **Feature Engineering:** Constructing relevant input features for prediction.
- **Deep Learning Model:** Implementing multiple neural network architectures to forecast stock returns.
- **Performance Evaluation:** Using Mean Squared Error (MSE) and Mean Absolute Error (MAE) to assess model accuracy.

## Methodology
### Data
- The dataset consists of **historical stock prices** of AAPL.
- Features include **closing prices, volume, and technical indicators**.
- A **train-test split** ensures unbiased evaluation.

### Machine Learning Approach
- **Feature Engineering:**
  - Log returns and price-based indicators.
  - Moving averages and volatility metrics.
- **Neural Network Models:**
  - **Feedforward Neural Network (FNN):** A fully connected deep learning model optimized with dropout and batch normalization to prevent overfitting. Performance showed limited improvement over traditional regression models, struggling to capture time dependencies effectively.
  - **Long Short-Term Memory (LSTM):** A recurrent neural network designed to capture sequential dependencies in stock price data. It significantly outperformed FNN in terms of capturing time-series patterns, making it the best standalone model for return prediction.
  - **Convolutional Neural Network (CNN):** Applied to extract spatial relationships in time-series stock data. While CNN was effective in capturing short-term patterns, it lacked the ability to model longer-term dependencies, leading to suboptimal performance compared to LSTM.
  - **Hybrid CNN-LSTM Model:** A combination of CNN and LSTM, leveraging CNN’s pattern recognition and LSTM’s sequential learning ability. This model delivered the best results overall, effectively balancing short-term and long-term prediction capabilities.
  - **Attention Mechanism with LSTM:** Implemented to enhance the focus on relevant time steps in sequential data. The attention-enhanced LSTM model outperformed standard LSTM in capturing key movements in stock prices, providing better interpretability and performance stability.
- **Evaluation Metrics:**
  - **MSE (Mean Squared Error):** Penalizes larger errors significantly.
  - **MAE (Mean Absolute Error):** Measures average error magnitude.

## Results & Performance
- The trained models demonstrated **predictive potential** for stock returns.
- **LSTM outperformed FNN and CNN**, capturing time-series dependencies more effectively.
- The **hybrid CNN-LSTM model achieved the best performance**, showing strong predictive power across multiple time horizons.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Matplotlib
- **Development Environment:** Jupyter Notebook

## Future Improvements
- Enhancing model **hyperparameter tuning** for improved performance.
- Incorporating **macroeconomic indicators** for broader market influence.
- Refining **ensemble learning techniques** to improve generalization.
- Further optimizing **attention mechanisms** to enhance interpretability and predictive power.
