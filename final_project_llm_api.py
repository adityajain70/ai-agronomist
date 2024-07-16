import google.generativeai as genai
import json

def setup_model():
    # get api key
    with open('./final-project-config.json') as f:
        config = json.load(f)
    GEMINI_API_KEY = config['gemini_api_key']
    genai.configure(api_key=GEMINI_API_KEY)

    # Set up the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    ]

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                generation_config=generation_config,
                                safety_settings=safety_settings)
    return model

def get_model_response(model, prompt):
    response = model.generate_content(prompt)
    return response.text

def get_management_strategy_recommendation(prompt):
    model = setup_model()
    management_strategy = get_model_response(model, prompt)
    return management_strategy

if __name__ == "__main__":
    get_management_strategy_recommendation()

