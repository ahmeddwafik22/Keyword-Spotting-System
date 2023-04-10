from prepare_data import DATASET_PATH,JSON_PATH,preprocess_dataset
from train import train,build_model,plot_history,EPOCHS,BATCH_SIZE,LEARNING_RATE,prepare_dataset,DATA_PATH,PATIENCE

preprocess_dataset(DATASET_PATH, JSON_PATH)
X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(DATA_PATH)
input_shape = (X_train.shape[1], X_train.shape[2], 1)
model = build_model(input_shape, learning_rate=LEARNING_RATE)
history = train(model, EPOCHS, BATCH_SIZE, PATIENCE, X_train, y_train, X_validation, y_validation)
plot_history(history)
test_loss, test_acc = model.evaluate(X_test, y_test)
print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))
