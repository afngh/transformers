import pickle

with open('config.pkl', 'rb') as f:
    config_data = pickle.load(f)

print(config_data.locales)