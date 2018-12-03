import requests
import json
import csv
import numpy as np
import pandas as pd

APP_ID = '4c5f3ddb'

API_KEY = '43600b1cc8bac1409d566f5613e43803'

API_ENDPOINT = 'https://api.edamam.com/api/nutrition-details?app_id=$APP_ID&app_key=$API_KEY'

file = pd.read_csv('C:\Vinay\Python_Deep_Learning\Project\Python_Project_Food\\results.csv')
recipe_names = []


with open('C:\Vinay\Python_Deep_Learning\Project\Python_Project_Food\\results.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file,delimiter=',')
    line_count = 0
    #print(csv_reader)
    for row in csv_reader:
      recipe_names.append(f'{row["Predictions"]}')


#url_one = "https://api.edamam.com/search?q="+i+"&app_id=dbc27b05&app_key=3a867d774216ecb55eba2387ecc3374d"


'''
&from=0&to=3
'''
for i in recipe_names:
    #f = open("nutrition_output.txt", 'w')
    print("For",i)
    #f.write(print("For", i))
    url_one = "https://api.edamam.com/search?q="+i+"&app_id=dbc27b05&app_key=3a867d774216ecb55eba2387ecc3374d"
    response = requests.get(url_one+"&app_id=dbc27b05&app_key=3a867d774216ecb55eba2387ecc3374d")
    data = response.json()
#print(type(data))
#print(data)

    hits = data["hits"]
#print(hits)
#print("\n")
    recepie = hits[0]
#print(recepie)
    recipe = recepie["recipe"]
#print(recipe)


    ingredients = recipe["ingredientLines"]
#print(json.dumps(ingredients))
#print(type(str))
#print(type(ingr))
    label = recipe["label"]

    print("Recipe is: ",label)
    #f.write(print("Recipe is: ", label))

#print("\n",ingredients, "\n")
#print(label)
#ingr = json.dumps(ingredients)
    ingr = ingredients

    print("Ingredients are: ",ingr)
    #f.write(print("Ingredients are: ", ingr))
    data_second = {'title':label,
                    'ingr':ingr}

#print(json.dumps(data_second, indent=4))

    url = "https://api.edamam.com/api/nutrition-data?app_id=4c5f3ddb&app_key=43600b1cc8bac1409d566f5613e43803"

    req = requests.get(url,data_second)

#print(req.status_code)
#print(req.text)
    inventory = json.loads(req.text)
#print(inventory)


    for key in inventory:
        if key == 'calories':
            print(key,inventory[key])
            #f.write(print(key, inventory[key]))

        if key == 'totalNutrients':
            print(key,inventory[key])
            #f.write(print(key, inventory[key]))
        #for key_two in inventory[key]


    print("-----------------------------------")