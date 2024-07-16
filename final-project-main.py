import requests
import json
import sys
from io import StringIO
import pandas as pd
import joblib
import xgboost as xgb
from final_project_llm_api import get_management_strategy_recommendation
import markdown
import webbrowser

def get_user_input():
    # Prompt user for input
    location = input("Enter your location as \"City, State\". (ie - Lincoln, Nebraska): ")
    nitrogen_level = float(input("Enter Nitrogen content in soil (kg/ha): "))
    phosphorus_level = float(input("Enter Phosphorus content in soil (kg/ha): "))
    potassium_level = float(input("Enter Potassium content in soil (kg/ha): "))
    
    return location, nitrogen_level, phosphorus_level, potassium_level

def get_environment_data(location):
    # get api key
    with open('./final-project-config.json') as f:
        config = json.load(f)
    WEATHER_API_KEY = config['visual_crossing_api_key']
    # define range to get environment data from
    start_date = '2023-04-01'
    end_date = '2023-09-30'
    average_temp = 0
    average_humidity = 0
    precipitation = 0
    # build api query
    url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start_date}/{end_date}?unitGroup=metric&include=days&key={WEATHER_API_KEY}&contentType=csv&elements=datetime,temp,humidity,precip'
    response = requests.get(url)
    
    # check for successful status code
    if response.status_code == 200:
        output_csv_file = StringIO(response.text)
        output_df = pd.read_csv(output_csv_file)
        # print(f'Average Temperature: {output_df["temp"].mean()}, Average Precipitation: {output_df["precip"].sum()}, Average Humidity: {output_df["humidity"].mean()}')
        average_temp = output_df['temp'].mean()
        average_humidity = output_df['humidity'].mean()
        precipitation = output_df['precip'].sum()
    elif response.status_code == 429: 
        print(f'Error: Too many requests. Please wait and try again later.\nRequest failed with status code {response.status_code}')
        sys.exit()
    else:
        print(f"Error: Request failed with status code {response.status_code}")
        sys.exit()

    return average_temp, average_humidity, precipitation

def get_soil_type_prediction(n, p, k, temperature, humidity, precip):
    # preprocess input data
    model_input = pd.DataFrame({'N_SOIL': [n], 'P_SOIL': [p], 'K_SOIL': [k],
                                'TEMPERATURE': [temperature], 'HUMIDITY': [humidity], 'RAINFALL': [precip]})
    model = joblib.load('./soil_type_prediction_model.pkl')
    try:
        prediction = model.predict(model_input)
        return prediction[0]
    
    except Exception as e:
        return f'An error occured during soil type prediction: {e}'

def get_crop_prediction(n, p, k, temperature, humidity, precip):
    # preprocess input data
    model_input = pd.DataFrame({'N': [n], 'P': [p], 'K': [k],
                                'temperature': [temperature], 'humidity': [humidity], 'rainfall': [precip]})
    model = joblib.load('./crop_prediction_model.pkl')    
    try:
        prediction = model.predict(model_input)
        return prediction[0]
    
    except Exception as e:
        return f'An error occured during best crop prediction: {e}'

def get_fertilizer_prediction(n, p):
    FERTILIZER_LABELS = ['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP', 'Urea']
    # preprocess input data
    model_input = pd.DataFrame({'Nitrogen': [n], 'Phosphorous': [p]})
    xgb_model_input = xgb.DMatrix(model_input)
    model = xgb.Booster()
    model.load_model('./xgboost_fertilizer_prediction_model.bin')
    try:
        prediction = model.predict(xgb_model_input)
        fertilizer_prediction = FERTILIZER_LABELS[int(prediction[0])]
        return fertilizer_prediction
    
    except Exception as e:
        return f'An error occured during fertilizer prediction: {e}'
    
def build_prompt(soil_type, crop, fertilizer, temperature, humidity, precip):
    prompt = [
        "Context:\nAct like an agronomist who specializes in advising farmers on crop management strategy and techniques to get the best yield. A farmer comes to you with specific details about their farm including their soil type (clay, loam, or sandy), crop they cultivate, fertilizer used ('Urea', 'DAP', '14-35-14', '28-28', '17-17-17', '20-20', or '10-26-26'), average temperature (Celsius), average relative humidity in %, and precipitation (in mm).\n\nObjective: \n\nCreate a plan for the best-suited crop management plan given the specific parameters and conditions given by the farmer. Outputted crop management strategy should be specific to the farmer's input and must include the following strategies: Seedbed Preparation, Planting schedule, Fertilization strategy, Pest Management strategy, Irrigation strategy, Harvesting schedule, and Post-Harvest Storage.\n\nStyle: \nDetailed, informational\n\nAudience: \nA farmer looking to maximize yield.\n\nRequirements: \nOutputted crop management strategy must include Seedbed Preparation, Planting schedule, Fertilization strategy, Pest Management strategy, Irrigation strategy, Harvesting schedule, and Post-Harvest Storage.",
        "input: Generate a crop management plan for a farmer who gives you the following details about their farm : \nSoil type = Clay soil, \nCrop they cultivate = Rice, \nFertilizer used = 14-35-14, \naverage temperature =  20 C, \naverage relative humidity = 82%, \naverage precipitation = 200 mm.",
        "output: Crop Management Plan for Rice Cultivation in Clay Soil\n\n1. Seedbed Preparation:\n- Begin seedbed preparation well in advance of planting season to ensure optimal soil conditions for rice germination.   \n- Plow the field thoroughly to break up soil clumps and create a fine tilth, which promotes seedling emergence.   \n- Incorporate organic matter such as compost or well-rotted manure into the soil to improve its structure and fertility.   \n- Level the seedbed evenly to facilitate uniform irrigation and minimize waterlogging.\n\n2. Planting Schedule:\n- Rice is typically planted in the wet season, taking advantage of ample rainfall and high humidity.   \n- Aim to plant rice when soil temperature reaches around 20-25°C, which is ideal for germination and early growth.   \n- Consider local climate patterns and aim to plant before the onset of heavy rains to avoid waterlogging.\n\n3. Fertilization Strategy:   \n- Utilize the fertilizer 14-35-14 to provide essential nutrients for rice growth and development.   \n- Apply fertilizer at the recommended rate based on soil test results and crop nutrient requirements.   \n- Incorporate fertilizer into the soil during seedbed preparation or apply as top dressing during the early stages of rice growth.   \n- Monitor plant nutrient status throughout the growing season and adjust fertilization as needed to optimize yield.\n\n4. Pest Management Strategy:  \n- Implement integrated pest management (IPM) practices to control pests while minimizing environmental impact.   \n- Monitor fields regularly for signs of pest infestation, focusing on common rice pests such as stem borers, leaf folders, and armyworms.   \n- Employ cultural practices such as crop rotation and maintaining proper water management to reduce pest pressure.   \n- Use biological control agents and biopesticides when feasible, supplementing with chemical pesticides only as a last resort and following label instructions carefully.\n\n5. Irrigation Strategy:\n- Maintain optimal soil moisture levels throughout the rice growing season, especially during critical growth stages such as tillering, flowering, and grain filling.   \n- Utilize flood or furrow irrigation methods suitable for clay soil to ensure uniform water distribution and minimize water loss through runoff or percolation.   \n- Schedule irrigation based on soil moisture monitoring and local weather forecasts, adjusting frequency and duration as needed to prevent water stress or waterlogging.\n\n6. Harvesting Schedule: \n- Monitor rice fields regularly for signs of maturity, including changes in color and texture of rice grains.   \n- Plan harvesting to coincide with optimal grain moisture content, typically around 20-25% for long-grain rice varieties.   \n- Use appropriate harvesting equipment such as combine harvesters or hand tools, taking care to minimize grain damage during the harvesting process.\n\n7. Post-Harvest Storage:\n- Dry harvested rice to reduce moisture content to safe levels (around 12-14%) to prevent mold growth and insect infestation during storage.   \n- Store dried rice in clean, airtight containers or silos to protect against moisture ingress and pest damage.   \n- Monitor stored rice regularly for signs of spoilage and implement proper ventilation and pest control measures as needed to maintain grain quality.",
        "input: Generate a crop management plan for a farmer who gives you the following details about their farm : \nSoil type = Loam soil, \nCrop they cultivate = Coffee, \nFertilizer used = Urea, \naverage temperature =  27 C, \naverage relative humidity = 66%, \naverage precipitation = 177 mm.",
        "output: Crop Management Plan for Coffee Cultivation in Loam Soil\n\n1. Seedbed Preparation: \n- Begin seedbed preparation several months before planting to ensure optimal soil conditions for coffee seedlings.   \n- Till the soil to a depth of 15-20 cm to loosen compaction and improve drainage, essential for coffee root development.   \n- Incorporate organic matter such as compost or aged manure to enhance soil fertility and structure.   \n- Create raised beds or mounds to improve drainage and prevent waterlogging, which can be detrimental to coffee plants.\n\n2. Planting Schedule:\n- Coffee is typically planted during the rainy season to take advantage of adequate soil moisture for establishment.   \n- Aim to plant coffee when temperatures are consistently warm, around 25-30°C, which promotes rapid root and shoot growth.   \n- Consider local climate patterns and aim to plant before the onset of heavy rains to allow for root establishment without waterlogging.\n\n3. Fertilization Strategy:\n- Utilize urea fertilizer to provide nitrogen, an essential nutrient for coffee plant growth and development.   \n- Apply urea fertilizer in split doses throughout the growing season to ensure continuous nutrient availability.   \n- Base fertilizer application rates on soil test results and crop nutrient requirements, adjusting as needed based on plant growth and visual symptoms of nutrient deficiency.\n\n4. Pest Management Strategy:\n- Implement a comprehensive pest management program to protect coffee plants from common pests such as coffee berry borer, leaf rust, and nematodes.   \n- Monitor coffee plants regularly for signs of pest infestation, including damaged leaves, berries, or wilting.   \n- Utilize cultural practices such as shade management and pruning to create a favorable environment for natural pest predators.   \n- Employ biological control agents and organic pesticides when possible, minimizing reliance on synthetic chemicals to reduce environmental impact.\n\n5. Irrigation Strategy:\n- Maintain optimal soil moisture levels throughout the coffee growing season, balancing water requirements with soil drainage characteristics.   \n- Utilize drip irrigation or sprinkler systems to deliver water directly to the root zone, minimizing water waste and evaporation.   \n- Monitor soil moisture levels regularly and adjust irrigation frequency and duration based on weather conditions and plant needs.   \n- Avoid overwatering to prevent waterlogging, which can lead to root rot and reduced crop yield.\n\n6. Harvesting Schedule:\n- Monitor coffee plants for signs of berry ripeness, including color changes from green to red or yellow depending on the variety.   \n- Plan harvesting to coincide with peak ripeness to ensure optimal flavor and quality of coffee beans.   \n- Utilize selective picking methods to harvest only ripe berries, leaving unripe berries on the plant to mature for later harvest.   \n- Employ trained laborers to handpick coffee berries carefully, minimizing damage to plants and ensuring high-quality beans.\n\n7. Post-Harvest Processing and Storage:\n- Immediately after harvest, process coffee berries to remove pulp and extract beans for drying.   \n- Utilize wet or dry processing methods based on local practices and environmental conditions.   \n- Dry coffee beans to the appropriate moisture content (10-12%) to prevent mold growth and ensure long-term storage stability.   \n- Store dried coffee beans in cool, dry conditions in breathable bags or containers to preserve flavor and freshness.",    
    ]
    prompt_task = f"input: Generate a crop management plan for a farmer who gives you the following details about their farm : \nSoil type = {soil_type}, \nCrop they cultivate = {crop}, \nFertilizer used = {fertilizer}, \naverage temperature =  {temperature} C, \naverage relative humidity = {humidity}%, \naverage precipitation = {precip} mm."
    prompt.append(prompt_task)
    prompt.append("output: ",)

    return prompt

def display_output(markdown_text):
    # Convert Markdown to HTML
    html = markdown.markdown(markdown_text)
    # Write HTML to a temporary file
    with open('./output.html', 'w') as f:
        f.write(html)
    webbrowser.open('output.html')

def main():
    # Get user input
    location, nitrogen_level, phosphorus_level, potassium_level = get_user_input()
    temperature, humidity, precipitation = get_environment_data(location)
    soil_type = get_soil_type_prediction(nitrogen_level, phosphorus_level, potassium_level, temperature, humidity, precipitation)
    print(f'Predicted soil type: {soil_type}')
    predicted_crop = get_crop_prediction(nitrogen_level, phosphorus_level, potassium_level, temperature, humidity, precipitation)
    print(f'Predicted crop to cultivate: {predicted_crop}')
    recommended_fertilizer = get_fertilizer_prediction(nitrogen_level,phosphorus_level)
    print(f'Recommended fertilizer: {recommended_fertilizer}')
    management_strategy_llm_prompt = build_prompt(soil_type, predicted_crop, recommended_fertilizer, temperature, humidity, precipitation)
    management_strategy_recommendation = get_management_strategy_recommendation(management_strategy_llm_prompt)
    print(management_strategy_recommendation)
    display_output(management_strategy_recommendation)
    
    
if __name__ == "__main__":
    main()
