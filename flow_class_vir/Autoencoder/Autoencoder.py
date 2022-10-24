import pandas as pd
from sklearn import preprocessing
import torch
import numpy as np
import os
import json
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
class MyDataset(Dataset):
 
  def __init__(self,set_values, labels):  
    x_set=pd.DataFrame.to_numpy(set_values)
    labels=np.array(labels)
    
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
    #y_test=le.transform(y_test)

    self.x_set=torch.tensor(x_set,dtype=torch.float32)
    self.y_set=torch.tensor(targets,dtype=torch.float32)
    self.le=le
  def __len__(self):
    return len(self.y_set)
   
  def __getitem__(self,idx):
    return self.x_set[idx],self.y_set[idx]


class AE_1layer(nn.Module):
	def __init__(self,n_hidden, **kwargs):
		super().__init__()
		self.encoder_hidden_layer = nn.Linear(
			in_features=kwargs["input_shape"], out_features=n_hidden
		)
		self.encoder_output_layer = nn.Linear(
			in_features=n_hidden, out_features=2
		)
		self.decoder_hidden_layer = nn.Linear(
			in_features=2, out_features=n_hidden
		)
		self.decoder_output_layer = nn.Linear(
			in_features=n_hidden, out_features=kwargs["input_shape"]
		)

	def forward(self, features):
		activation = self.encoder_hidden_layer(features)
		activation = torch.relu(activation)
		code = self.encoder_output_layer(activation)
		code = torch.relu(code)
		activation = self.decoder_hidden_layer(code)
		activation = torch.relu(activation)
		activation = self.decoder_output_layer(activation)
		#reconstructed = torch.relu(activation)
		reconstructed=activation
		return reconstructed
	
	def codings(self, features):
		activation = self.encoder_hidden_layer(features)
		activation = torch.relu(activation)
		code = self.encoder_output_layer(activation)
		return code

class AE_2layer(nn.Module):
    def __init__(self,n_hidden_1, n_hidden_2, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=n_hidden_1
        )
        self.encoder_hidden_layer_2 = nn.Linear(
            in_features=n_hidden_1, out_features=n_hidden_2
        )
        self.encoder_output_layer = nn.Linear(
            in_features=n_hidden_2, out_features=2
        )
        self.decoder_hidden_layer_2 = nn.Linear(
            in_features=2, out_features=n_hidden_2
        )
        self.decoder_hidden_layer_1 = nn.Linear(
            in_features=n_hidden_2, out_features=n_hidden_1
        )
        self.decoder_output_layer = nn.Linear(
            in_features=n_hidden_1, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)

        activation = self.encoder_hidden_layer_2(activation)
        activation = torch.relu(activation)

        code = self.encoder_output_layer(activation)
        code = torch.relu(code)

        activation = self.decoder_hidden_layer_2(code)
        activation = torch.relu(activation)

        activation = self.decoder_hidden_layer_1(activation)
        activation = torch.relu(activation)

        activation = self.decoder_output_layer(activation)
        #reconstructed = torch.relu(activation)
        reconstructed=activation
        return reconstructed
    
    def codings(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)

        activation = self.encoder_hidden_layer_2(activation)
        activation = torch.relu(activation)

        code = self.encoder_output_layer(activation)
        return code
		
def load_model(n_layers,n_hidden_1, n_hidden_2):
  """
  Loads the model information

  n_layers: int
	number of layers

  n_hidden_1: int
	number of units in second layer

  n_hidden_2: int
	number of units in second layer
  """
  if n_layers==1:
    pathname=str(n_layers)+"Layer_"+"nunits1_"+str(n_hidden_1)
    pathname=os.path.join("models",pathname)
    model = AE_1layer(n_hidden=n_hidden_1,input_shape=7)

  if n_layers==2:
    pathname=str(n_layers)+"Layer_"+"nunits1_"+str(n_hidden_1)+"_"+"nunits2_"+str(n_hidden_2)
    pathname=os.path.join("models",pathname)
    model = AE_2layer(n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2, input_shape=7)

  # Load model
  filename="model.pth"
  filename=os.path.join(pathname,filename)
  model.load_state_dict(torch.load(filename))

  # load foldperf
  filename="information.json"
  filename=os.path.join(pathname,filename)
  with open(filename, 'r') as file:
	  foldperf=json.load(file)


  # load batch_loss
  filename="batch_loss.npz"
  filename=os.path.join(pathname,filename)
  #np.load(filename).files
  batch_loss=np.load(filename)["batch_loss"]
  return batch_loss, foldperf, model

def Cross_val_loss(foldperf, k=4):
  """
  Gives the average loss of validation set in each fold

  foldperf: dictionary 
	Loss of each validation folder and epoch

  k: int
	number of folders
  """
  vall_f=[]
  for f in range(1,k+1):
	  vall_f=np.append(vall_f,np.mean(foldperf['fold{}'.format(f)]['val_loss']))

  return vall_f

def test_loss(model, X_test):
  """
  Calculates the loss in the test set

  model: model class
	Model of the autoencoder
  X_test: torch.FloatTensor
	set of data with test instances
  """
  # mean-squared error loss
  criterion = nn.MSELoss()
  with torch.no_grad():
		  X_output = model.forward(X_test)
  return criterion(X_test,X_output)
