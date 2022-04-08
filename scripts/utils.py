import toml


def load_toml(toml_path: str):
    with open(toml_path) as f:
        dict_toml = toml.load(f)
    return dict_toml
