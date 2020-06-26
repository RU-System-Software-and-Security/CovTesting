import numpy as np

# the data is in range(-.5, .5)
def load_data(name):
    assert (name.upper() in ['MNIST', 'CIFAR', 'SVHN'])
    name = name.lower()
    x_train = np.load('../data/' + name + '_data/' + name + '_x_train.npy')
    y_train = np.load('../data/' + name + '_data/' + name + '_y_train.npy')
    x_test = np.load('../data/' + name + '_data/' + name + '_x_test.npy')
    y_test = np.load('../data/' + name + '_data/' + name + '_y_test.npy')
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    dataset = 'mnist'
    model_name = 'lenet1'
    l = [0, 8]

    x_train, y_train, x_test, y_test = load_data(dataset)

    # ## load mine trained model
    from keras.models import load_model

    model = load_model('../data/' + dataset + '_data/model/' + model_name + '.h5')
    model.summary()

    index = np.load('fuzzing/nc_index_test_{}.npy'.format(0), allow_pickle=True).item()
    for y, x in index.items():
        print(y)
        x_test[y] = x

    index = np.load('fuzzing/nc_index_test_{}.npy'.format(1), allow_pickle=True).item()
    for y, x in index.items():
        print(y)
        x_test[y] = x

    np.save('x_test_new.npy', x_test)
