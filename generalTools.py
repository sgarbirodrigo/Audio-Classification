import pickle


def save_dict(obj, name):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, 0)


def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
