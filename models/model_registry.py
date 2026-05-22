class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, model_name, model_class):
        self.models[model_name] = model_class

    def get_model_class(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} is not registered.")
        return self.models[model_name]

    def get_all_model_names(self):
        return list(self.models.keys())


model_registry = ModelRegistry()
