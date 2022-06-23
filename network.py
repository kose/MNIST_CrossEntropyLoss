import torch
from torch import nn
from torch.nn import functional as F

dropout1 = 0.31546562430950537
dropout2 = 0.22604990800031521
lr = 0.0009169554556810893
n_cnn1 = 47
n_cnn2 = 28
n_ln = 577

class CNN(nn.Module):

    def __init__(self, kernel_size=5, n_cnn1=n_cnn1, n_cnn2=n_cnn2, n_ln=n_ln, dropout1=dropout1, dropout2=dropout2):
        
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, n_cnn1, kernel_size=kernel_size, stride=1, padding=0)
        self.conv2 = nn.Conv2d(n_cnn1, n_cnn2, kernel_size=kernel_size, stride=1, padding=0)

        if kernel_size == 3:
            n_ln0 = int(n_cnn2 * 12 * 12)

        if kernel_size == 5:
            n_ln0 = int(n_cnn2 * 10 * 10)
            
        self.fc1 = nn.Linear(n_ln0, n_ln)
        self.fc2 = nn.Linear(n_ln, 10)

        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

        
    def forward(self, x):
        batch_size = x.shape[0]

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 2)
        h = self.dropout1(h)

        h = torch.flatten(h, 1)

        h = F.relu(self.fc1(h))
        h = self.dropout2(h)

        return torch.sigmoid(self.fc2(h))

        
# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
