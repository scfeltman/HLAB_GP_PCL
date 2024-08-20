import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import pandas as pd
import openpyxl
import numpy as np
from sklearn.metrics import mean_absolute_error

%matplotlib inline
%load_ext autoreload
%autoreload 2

PCLdat = pd.read_excel("PCLscores.xlsx")

def closeAbove(mylist,target):

    if target > 90:
        return "bruh wut"
    
    try:
        x = mylist.index(target)
    except:
        x = closeAbove(mylist,target+1)

    return x

def createSplit(day_list,pcl_list,start):
    try:
        day_list.index(start+60) == -1 or day_list.index(start+61) == -1 or day_list.index(start+66) == -1
    except:
        return []
    else:

        testindex = [day_list.index(start+60), day_list.index(start+61) ,day_list.index(start+66)]
        i_end = day_list.index(start+60)-1
        i_start = closeAbove(day_list,start)

        
        day_extract = torch.tensor(day_list[i_start:(i_end+1)])
        pcl_extract = torch.tensor(pcl_list[i_start:(i_end+1)])

        day_test = torch.tensor([day_list[a] for a in testindex])
        pcl_test = torch.tensor([pcl_list[a] for a in testindex])

        if len(day_extract) != len(pcl_extract) or len(day_test)+len(pcl_test) != 6:
            raise Exception("Arrays are not the same size.")
        
        X = (day_extract,day_test)
        Y = (pcl_extract,pcl_test)

        return (X,Y)

traincount = 163 ## 65%
devcount = 25 ## 10%
testcount = 63 ## 25%

train_set = list()
dev_set = list()
test_set = list()

ids = PCLdat["PID"].unique().tolist()

for ii in range(traincount):
    temp = PCLdat[PCLdat["PID"] == ids[ii]]
    tempday = temp["Day"].tolist()
    temppcl = temp["PCL"].tolist()
    for jj in range(1,46):
        if jj >= len(tempday) - 66:
            break
        res = createSplit(tempday,temppcl,jj)
        if len(res) != 0:
            train_set.append((res,ids[ii]))

class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

train_batch = np.random.randint(0,len(train_set),size=30)

training_iter = 100
n1 = []
n2 = []
n7 = []
f, ax = plt.subplots(6, 5, figsize=(35, 25))
for ii in range(len(train_batch)):
    currID = train_set[train_batch[ii]][1]
    train_x = train_set[train_batch[ii]][0][0][0]
    train_y = train_set[train_batch[ii]][0][1][0]
    test_x = train_set[train_batch[ii]][0][0][1]
    test_y = train_set[train_batch[ii]][0][1][1]
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SpectralMixtureGPModel(train_x, train_y, likelihood)
    torch.manual_seed(123)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(torch.cat([train_x,test_x])))
        n1.append(np.abs(observed_pred.mean[-3] - test_y[0]))
        n2.append(np.abs(observed_pred.mean[-2] - test_y[1]))
        n7.append(np.abs(observed_pred.mean[-1] - test_y[2]))
        lower, upper = observed_pred.confidence_region()
        ax[ii%6,ii//6].plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax[ii%6,ii//6].plot(torch.cat([train_x,test_x]).numpy(), observed_pred.mean.numpy(), 'b')
        ax[ii%6,ii//6].plot(test_x.numpy(), test_y.numpy(), 'r*')
        # Shade between the lower and upper confidence bounds
        ax[ii%6,ii//6].fill_between(torch.cat([train_x,test_x]).numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax[ii%6,ii//6].set_ylim([torch.min(torch.cat([train_y,test_y])).tolist()-1, torch.max(torch.cat([train_y,test_y])).tolist()+1])
        ax[ii%6,ii//6].legend(['Train data', 'Mean', 'Test data', 'Confidence'])
        ax[ii%6,ii//6].set_title("P" + str(currID) + " MAE: " + str(mean_absolute_error(torch.cat([train_y,test_y]),observed_pred.mean)))

f.suptitle("PCL trajectory predictions - a random selection",fontsize = 24)
f.savefig("PCLplots.pdf", format="pdf")