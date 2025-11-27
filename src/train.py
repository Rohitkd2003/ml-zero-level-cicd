from model_utils import EvenOddModel

def train_and_save():
    model = EvenOddModel()
    model.train()
    model.save_model("model.pkl")
    print("Model trained and saved!")

if __name__ == "__main__":
    train_and_save()
