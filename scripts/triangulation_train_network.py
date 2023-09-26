#imports
import torch
import os,sys, math

os.chdir(os.path.dirname(__file__))
sys.path.append('../src/models')
sys.path.append('../src/problem_generators')
import triangulation_model_v0 as model_spec
from generate_triangulation_problem import package_problems


# config
nr_max = 30
noise_std = 0.1
outlier_percentage = 0.5
batch_size = 64
n_layers = 4
n_heads = 6 # Need to be divisible by 3 currently
demb = 64*n_heads
device = 'mps' if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
epochs = 5000
lr = 3e-4





#initialize
model = model_spec.TransformerNetwork(n_layers,n_heads,demb).to(device)
model.to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

#model = torch.load("model_v5_80")
# Training loop
def train(model, loss_fn, optimizer, batch_size, batches_in_epoch):
    model.train()
    btop = 20
    loss_summer = 0
    
    for batch in range(batches_in_epoch):
        
        X,y = package_problems(batch_size, nr_max,noise_std=noise_std,outlier_percentage=outlier_percentage)
        
        X = X.to(device)
        y = y.to(device)

        pred = model(X)

        pred = torch.nan_to_num(pred)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_summer += loss.item()/btop
        
        
        if batch % btop == btop-1:
            
            loss = loss_summer

            loss_summer = 0
            print(f"loss: {math.sqrt(loss):>7f}  [{batch:>5d}/{batches_in_epoch}]")


for t in range(epochs):
    if t % 200 == 0:
        torch.save(model, "model_v5_" + str(t))
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, loss_fn, optimizer, batch_size, 100)
print("Done!")