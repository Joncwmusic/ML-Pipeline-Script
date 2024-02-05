import numpy as np
import pandas as pd
import CleaningFunctions as cf
import random


# pd.set_option()
# Generate data with missing values, Categorical Variables, and crazy outliers
weight_list = [random.randint(140, 270) for i in range(500)]
height_list = [random.randint(54, 90) for item in weight_list]
gender_list = [random.randint(0, 1) for k in weight_list]
activity_list = [random.randint(0, 10) for l in weight_list]
sport_list = [random.choice(["weightlifting", "baseball", "soccer", "running", "swimming"]) for m in activity_list]
bmr_list = []

for i in weight_list:
    if sport_list[i] in ["weightlifting", "baseball"]:
        bmr_list.append((1.1-0.2*random.random()) * (weight_list[i]*10 + height_list[i]*8 + gender_list[i]*270 + activity_list[i]*100))
    else:
        bmr_list.append((1.1-0.2*random.random()) * (weight_list[i]*9.5 + height_list[i]*8 + gender_list[i]*270 + activity_list[i]*200))

for item in weight_list:
    random_val = random.randint(0,100)
    if random_val in [4, 10, 13, 56, 77]:
        weight_list[random_val] = None
    if random_val in [5, 13, 34, 49, 92]:
        height_list[random_val] = None
    if random_val in [6, 13, 49, 79, 99]:
        activity_list[random_val] = None
    if random_val in [4, 6, 67, 79, 77]:
        sport_list[random_val] = None

# print(weight_list, height_list, activity_list, sport_list)
test_data = {"weight": weight_list, "gender": gender_list, "height" : height_list, "activity_level": activity_list, "Maintain_kCal" : bmr_list, "sport": sport_list}

unclean_data = pd.DataFrame(test_data)

print(unclean_data)

new_data0 = cf.replace_null_with_other(unclean_data, ['sport'])
new_data1 = cf.create_dummy_variables(new_data0, ['sport'])
new_data2 = cf.replace_null_with_average(new_data1, ['weight', 'height'])
new_data3 = cf.replace_null_with_zero(new_data2, ['activity_level'])


print(new_data3)