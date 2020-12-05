# Foof Type Detection and Calorie anagement (Python-DeepLearning-Project)

Food Image Dataset: https://www.dropbox.com/s/ps0b3kg6hm8iyq8/FOOD%20DATASET%20IMAGES.zip?dl=0     

# AUTHORS
This report contains all the documents for the Group project. The assignment was done by
Fatema Hasta (Class ID: 10), Vinay Jaibheem (Class ID: 11), Farid Uddin Ahmed (Class ID: 02)
and Zarin Tasnim Sandhie (Class ID: 26), all of them are students for the course Python/Deep
learning (CSEE5590/490, Fall 2018) at the University of Missouri-Kansas City (UMKC).

# OBJECTIVE
The objectives of this Project is:
* It will take input in the form of an image.
* This image can be fed to the model that detects the type of food and classifies the image
either into Cup Cake, Chicken wings, cheescake etc.
* Based on this category, particular feature will be appeared to the user.

# METHODOLOGY
* In order to detect the type of food, there will be a food dataset categorized into Cup Cake,
Chicken wings, cheescake etc. For our case,
* Convolutional Neural Networks (CNN) (Inception model) can be used to classify the new
image of a food into one of these classes.
* By utilizing Edamam Food API, nutrition details of the food is identified.
* The datasets can later be extended into a more elaborate dataset.

# CODE EXPLANATION
* For our project, we have taken 10 classes from “Food-101” dataset.
* The classes we have used are: ‘apple_pie', 'baklava', 'bread_pudding', 'carrot_cake', 'cheesecake', 'chicken_curry', 'chicken_wings', 'chocolate_cake', 'club_sandwich' and
'cup_cakes'.
* Each class has 1000 total pictures.
* We have created 2 different folders named “train” and “validation” each of those contain 10 folders for each classes.
* The train folder is used for training dataset and each of the class folders inside train folder contains 7500 pictures.
* The validation folder is used for validating the dataset and each of the class folders inside the validation folder contain 2500 pictures.
* The test folder contain 20 random pictures from different classes which are later used for testing the model.
* All the images are augmented at first in which procedure the pictures are rotated, shifted, whitened etc. to increase the number of data in our dataset.
* Then inception is used as a pre-trained model.
* For the optimization, two different types of optimizers are used in two different pass.
* For the first pass, we used RMSprop and 5 epochs. The optimization is run for the whole model.
* For the second pass, SGD optimizer is used and 10 epochs is used. Here, expect the last two layers, all the other layer are frozen by making it false. The optimization is done only for the last two layers.
* Then the model is saved.
* This model can later be loaded and tested with test images.
* When we predict the name of an unknown food class from the model, this predicted name is saved in a .csv file.
* Then we pass this food name from the csv file to Edamam Api.
* Edamam Api has two features: one is to extract recipe and another is to extract calorie and nutrition chart from the recipe.
* At first, we extracted the recipe and then using that recipe, we extracted the calorie and nutrition

