import kaggle

kaggle.api.authenticate()
competitions = kaggle.api.competitions_list()
print(competitions)