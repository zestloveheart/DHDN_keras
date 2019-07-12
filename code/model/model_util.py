from .DHDN import DHDN
from .DIDN import DIDN
from pathlib import Path
def model_getter(model_name,model_path="",summary = False):
    model_catalog = {"DHDN":DHDN,"DIDN":DIDN}
    assert model_name in model_catalog, "the model_name is not exist !"
    model = model_catalog[model_name]()
    if summary:
        model.summary()
    if model_path != "":
        assert Path(model_path).exists(),'can not load the model from the path, maybe is not exist'
        model.load_weights(model_path)
    return model