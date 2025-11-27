import os
from src.model_utils import EvenOddModel

def test_training():
    model = EvenOddModel()
    assert model.train() == True

def test_model_file_created():
    model = EvenOddModel()
    model.train()
    model.save_model("model.pkl")
    assert os.path.exists("model.pkl")

def test_predict_method():
    model = EvenOddModel()
    model.train()
    assert hasattr(model, "predict")
    assert model.predict(2) == 0
    assert model.predict(3) == 1
