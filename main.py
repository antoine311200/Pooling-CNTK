import matplotlib.pyplot as plt
from data import load_data

from sklearn.model_selection import train_test_split

from cntk import ConvNTK

if __name__ == "__main__":
    X, Y = load_data()
    # Keep only the first 1000 samples
    X = X[:100]
    Y = Y[:100]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print("X_train shape: ", X_train.shape)
    print("Y_train shape: ", Y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("Y_test shape: ", Y_test.shape)

    cntk = ConvNTK(depth=5, final_layer_name="vanilla", fix_layer=True, verbose=True)
    cntk.train(X_train, Y_train)
    accuracy = cntk.evaluate(X_test, Y_test)
    print("Accuracy: ", accuracy)