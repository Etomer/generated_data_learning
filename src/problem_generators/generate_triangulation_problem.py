import torch 

def generate_problem(nr, noise_std=0, outlier_percentage=0):
    sender_position = torch.rand(1,2)
    receiver_position = torch.rand(nr,2)

    # measuring distances
    measurements = torch.cdist(sender_position,receiver_position)

    #adding noise
    measurements = measurements + noise_std*torch.randn(1,nr)

    # adding outliers
    outliers = torch.rand(1,nr) < outlier_percentage
    measurements[outliers] = torch.rand(measurements[outliers].shape)*measurements.max()
    return (measurements, sender_position, receiver_position)

def package_problems(batch_size, nr_max, noise_std=0, outlier_percentage=0):
    
    X = torch.zeros(batch_size,nr_max,3)
    y = torch.zeros(batch_size,2)
    for i in range(batch_size):
        measurements, sender_position, receiver_position = generate_problem(nr_max,noise_std,outlier_percentage)
        #receiver_index = torch.arange(1).unsqueeze(1)@torch.ones(nr_max,dtype=torch.int64).unsqueeze(0)
        #sender_index = torch.ones(nr_max,dtype=torch.int64).unsqueeze(1)@torch.arange(ns_max).unsqueeze(0)
        #X[i,:] = torch.stack([measurements.flatten(),sender_index.flatten(),receiver_index.flatten()],dim=1)
        
        
        X[i,:] = torch.cat([measurements.T, receiver_position],dim=1)
        y[i,:] = sender_position
    return X,y