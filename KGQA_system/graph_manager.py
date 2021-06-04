import matplotlib.pyplot as plt


class MyGraph:
    def __init__(self):
        self.train_loss = []
        self.valid_loss = []
        self.positive_loss = []
        self.negative_loss = []
        self.lr = []
        self.average_train_loss = []
        self.hits_1 = []
        self.hits_3 = []
        self.hits_10 = []