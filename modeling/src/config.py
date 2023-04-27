import yaml

def load_config(filename='config/config.yml'):
    with open(filename, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg