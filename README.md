# ai-agronomist
This project aims to create an AI agronomist that predicts soil type, recommends the best crop to cultivate, suggests suitable fertilizers, and generates a detailed crop management strategy. It leverages Google's Gemini Large Language Model (LLM) and various machine-learning models to provide comprehensive farming recommendations.

## Motivation and Purpose

The AI Agronomist project was conceived to address several critical aspects of modern agriculture. By leveraging machine learning and AI, the project aims to:

- Improve crop yields.
- Reduce fertilizer costs and waste.
- Provide personalized agricultural recommendations tailored to specific environmental and soil conditions.
- Offer farmers valuable insights without the need for an in-person agronomist, thereby saving costs.
- Enhance sustainable farming practices through data-driven decision-making.

The ultimate goal is to equip farmers with the tools and knowledge they need to make informed decisions that optimize their agricultural outcomes.

## Project Structure

The project consists of the following key components:

- `final-project-main.py`: The main script that integrates all functionalities.
- `final_project_llm_api.py`: Handles the setup and interaction with Google's Gemini LLM for generating crop management strategies
- `final-project-weather-api.py`: Fetches weather data using the Visual Crossing Weather API.
- `final-project-model1.ipynb`, `final-project-model2.ipynb`, `final-project-model3.ipynb`: Jupyter notebooks for building and training the soil type prediction, crop prediction, and fertilizer recommendation models respectively.

## Data Sets Used

- **Precision Crop Optimization Dataset**: The dataset encapsulates 1000 unique agricultural scenarios with these attributes: District, Soil Type, pH Level, Organic Matter (%), Nitrogen Content (kg/ha), Phosphorus Content (kg/ha), and Potassium Content (kg/ha).
  - Source: [Kaggle - Precision Crop Optimization Dataset](https://www.kaggle.com/datasets/janhavisrivastava14/precision-crop-optimization-dataset)
- **Crop Recommendation Data**: This dataset was made by augmenting optimum soil and environmental characteristics for crop growth. 
  - Source: [Harvard Dataverse - Crop Recommendation Data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/4GBWFV)
- **Fertilizer Prediction**: This data includes fertilizers used in 100 unique agricultural settings. Fertilizer classes include Urea, DAP, and Compound (10-26-26, 14-35-14, 17-17-17, 20-20, 28-28).
  - Source: [Kaggle - Fertilizer Prediction](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction)

## Setup
Create a `final-project-config.json` file in the root directory with the following structure:

```json
{
  "gemini_api_key": "your_gemini_api_key",
  "visual_crossing_api_key": "your_visual_crossing_api_key"
}
```

Replace `your_gemini_api_key` and `your_visual_crossing_api_key` with your actual API keys.

## Example

Here is an example of how to run the project:

```bash
python final-project-main.py
```

Follow the prompts to input:

- Location: "Lincoln, Nebraska"
- Nitrogen content: 50 kg/ha
- Phosphorus content: 30 kg/ha
- Potassium content: 20 kg/ha

The script will output predictions for soil type, the best crop to cultivate, recommended fertilizer, and a detailed crop management strategy.
