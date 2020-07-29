import torch
import torch.nn as nn
from torch.nn import Conv1d, Conv2d, Sequential, ModuleList, BatchNorm2d
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from sklearn.metrics import confusion_matrix



class HARDataLoader:
    def __init__(self, data_root):
        self.data_root = data_root
        
    def pre_operation(self, **kwargs):
        pass
    
    def post_operation(self, **kwargs):
        X = kwargs.get("X", None)
        y = kwargs.get("y", None)
        return X, y
       

    # load a single file as a numpy array
    def load_file(self, filepath):
        dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
        return dataframe.values

    # load a list of files into a 3D array of [samples, features, timesteps]
    def load_group(self, filenames, prefix=''):
        loaded = list()
        for name in filenames:
            data = self.load_file(prefix + name)
            loaded.append(data)
        
        # stack group and transpose to (samples, features, timesteps)
        loaded = np.dstack(loaded).transpose(0,2,1)
        return loaded

    # load a dataset group, such as train or test
    def load_dataset_group(self, group):
        filepath = os.path.join(self.data_root, group, 'Inertial Signals/')
        # load all 9 files as a single array
        filenames = list()
        # total acceleration
        filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
        # body acceleration
        filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
        # body gyroscope
        filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
        # load input data
        X = self.load_group(filenames, filepath)
        # load class output
        y = self.load_file(os.path.join(self.data_root, group, 'y_'+group+'.txt'))
        return X, y

    # load the dataset, returns train and test X and y elements
    def load_dataset(self, **kwargs):
        
        train_test = kwargs.get("train_test", "train")
        
        self.pre_operation(**kwargs)
        
        # load all train
        X, y = self.load_dataset_group(train_test)
        
        # zero-offset class values
        y = y - 1
        
        X, y = self.post_operation(X=X, y=y)
        
        return X, y
    
    def load_labels(self):
        labels_file = os.path.join(self.data_root, 'activity_labels.txt')
        return list(pd.read_csv(labels_file, header=None, delim_whitespace=True)[1].array)

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]
    
    # standardize data
    def scale_data(self, trainX, testX, standardize):
        # remove overlap
        cut = int(trainX.shape[1] / 2)
        longX = trainX[:, -cut:, :]
        # flatten windows
        longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
        # flatten train and test
        flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
        flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
        # standardize
        if standardize:
            s = StandardScaler()
            # fit on training data
            s.fit(longX)
            # apply to training and test data
            longX = s.transform(longX)
            flatTrainX = s.transform(flatTrainX)
            flatTestX = s.transform(flatTestX)
        # reshape
        flatTrainX = flatTrainX.reshape((trainX.shape))
        flatTestX = flatTestX.reshape((testX.shape))
        return flatTrainX, flatTestX


class HARDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.Transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.y[index]
        if self.Transform:
            return self.Transform(x), y
        else:
            return x, y

    def __len__(self):
        return len(self.X)


class Experiment:
    def __init__(self, config):
        self.config = config
        self.best_predictions = torch.empty(0)
        self.actuals = torch.empty(0)
        self.best_score = 0.0
        self.best_model = None
        self.labels=[]
        self.verbose = config.get("verbose", False)
        
        
    '''
        Function to update the config.
        Should be called by the experimenter is config needs to be 
        updated between experiment runs.
    '''
    def update_config(self, config):
        self.config = config
    
    '''
        Trains the network for n epochs and returns an accuracy score
    '''
    def train_and_evaluate(self, train_loader, test_loader):
        
        # If a random seed is set, weights, etc are initialised consistently 
        # between different runs.
        random_seed = self.config.get("random_seed", 0)
        if random_seed > 0:
            torch.manual_seed(random_seed)
        
        net_class = self.config.get("net_class", None)
        
        # We need the class name of the network to create dynamically.
        if net_class is None:
            print("net_class not supplied.")
            return
        
        # Create the net object from the class name. 
        net = globals()[net_class](self.config)
        
        if self.config.get("init_weights", False):
            # Initialise the network's weights
            net.apply(self.init_weights)
        
        # Get all the config parameters
        lr = self.config.get("lr", 0.01)
        batch_size = self.config.get("batch_size", 32)
        epochs = self.config.get("epochs_per_repeat",10)
        
        # Create the optimiser
        optimiser = optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        train_size = len(train_loader.dataset)
        for epoch in range(epochs):
            
            # Training
            
            net.train()
            correct = 0
            for X, y in train_loader:
                optimiser.zero_grad()
                X, y = X.float(), y.long().squeeze()
                
                out = net(X)
                if len(out) == 2:
                    y_hat, logits = out
                else:
                    y_hat, logits = out, out
                    
                loss = loss_fn(logits, y)
                loss.backward()
                optimiser.step()
                predicted = torch.argmax(y_hat.data,1)
                correct += (predicted == y).sum()

            acc_train = float(correct) * 100.0 / train_size
            if self.verbose:
                print("Train accuracy:", acc_train, "Loss: ", loss.item())
            
            # Interim Evaluation
            if epoch % 10 == 0:
                if self.verbose:
                    accuracy, actuals, predictions = self.evaluate(net, test_loader)
                    print("Testing accuracy: ", accuracy)
                
        # Final Evaluation
        accuracy, actuals, predictions = self.evaluate(net, test_loader)
        if accuracy > self.best_score:
            self.best_score = accuracy
            self.best_predictions = predictions
            self.best_model = net
            self.actuals = actuals
        return accuracy.item()

    
    def evaluate(self, net, test_loader):
        net.train(False)
        predictions = torch.empty(0)
        actuals = torch.empty(0)
        total_test_size = 0
        with torch.no_grad():
            correct = 0
            for X, y in test_loader:
                X, y = X.float(), y.long().squeeze()

                out = net(X)
                if len(out) == 2:
                    y_hat, logits = out
                else:
                    y_hat, logits = out, out

                predicted = torch.argmax(y_hat.data,1)
                predictions = torch.cat((predictions.float(), predicted.float()),0) 
                actuals = torch.cat((actuals.float(), y.float()), 0)
                total_test_size += y.size(0)
                correct += (predicted.numpy() == y.numpy()).sum()
        accuracy = correct * 100. / total_test_size
        return accuracy, actuals, predictions

    
    def init_weights(self, m):
        if type(m) in (nn.Linear, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    
    '''
        summarize scores
    '''
    def summarize_results(self):
        m, s = np.mean(self.scores), np.std(self.scores)
        repeats = self.config.get("repeats", 10)
        print('Accuracy over {0} trials: {1:.3f}% (+/-{2:.3f})'.format(repeats, m, s))
        
        print('Best model accuracy: {0:.3f}%'.format(np.max(self.scores)))
        
        # confusion matrix for best scores
        conf_matrix = confusion_matrix(self.actuals, self.best_predictions)
        df_cm = pd.DataFrame(conf_matrix, index = self.labels, columns = self.labels)
        plt.figure(figsize = (3,3))
        plt.title('Confusion Matrix For Best Run', fontsize=20)
        sns.heatmap(df_cm, annot=True, fmt='g')
        plt.show()
        if len(self.scores) > 1:
            self._plot(self.scores)

    # helper function to plot scores
    def _plot(self, data):
        plt.figure(figsize = (3,3))
        sns.lineplot(range(1, len(data) + 1), data)
        
        plt.xlabel('Repeats', fontsize=14)
        plt.ylabel('Accuracy(%)', fontsize=14)
        plt.title('Test Accuracy', fontsize=20)
        plt.show()

    ''' 
        Run the experiment
    '''
    def run(self):
        
        # load data
        har_dataloader = HARDataLoader('../data/raw/UCI_HAR_Dataset')
        train_X, train_y = har_dataloader.load_dataset(train_test='train') 
        test_X, test_y = har_dataloader.load_dataset(train_test='test')
        self.labels = har_dataloader.load_labels()
        
        # scale data
        if self.config.get("standardize", False):
            train_X, test_X = har_dataloader.scale_data(train_X, test_X, standardize=True)

        # repeat experiment
        self.scores = list()
        repeats = self.config.get("repeats", 10)
        batch_size = self.config.get("batch_size", 32)
        for r in range(repeats):
            # Create the data loaders
            train_set = HARDataset(train_X, train_y)
            test_set = HARDataset(test_X, test_y)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

            # train and evaluate the model
            score = self.train_and_evaluate(train_loader, test_loader)
            print('#{0}: {1:.3f}'.format(r+1, score))
            self.scores.append(score)
        # summarize results
        self.summarize_results()

class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        
        self.config = config
        
        input_dim = self.config["input_dim"]
        output_dim = self.config["output_dim"]
        
        # Hidden dimensions
        self.hidden_dim = self.config["hidden_dim"]

        # Number of hidden layers
        self.layer_dim = self.config["layer_dim"]

        # Build the LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        
        # x is in shape (batch_sise, 9, 128)
        # need to transpose it to (batch_size, 128, 9)
        x = torch.transpose(x, 2, 1)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 6
        return out

if __name__ == "__main__":

    dl = HARDataLoader('../data/raw/UCI_HAR_Dataset')

    # Load the dataset
    train_X, train_y = dl.load_dataset(train_test='train') 
    test_X, test_y = dl.load_dataset(train_test='test')


    print("Training Data:", train_X.shape, train_y.shape)
    print("Testing Data:", test_X.shape, test_y.shape)

    # Create torch datasets and data loaders for train and test
    train_set = HARDataset(train_X, train_y)
    test_set = HARDataset(test_X, test_y)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

    print("Num Training samples: ", len(train_loader.dataset))
    print("Num Testing samples:", len(test_loader.dataset))

    config = dict(
    net_class = "LSTMModel",
    # network related config
    input_dim = 9, # number of rows in the input
    hidden_dim = 100,
    layer_dim = 1,
    output_dim = 6,
    # -----
    repeats = 1,
    epochs_per_repeat = 50,
    lr = 0.001,
    batch_size = 32
    )
    experiment = Experiment(config)
    experiment.run()

