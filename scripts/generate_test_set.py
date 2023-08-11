import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import torch

# Currently using only recordings with 12 mics, and using all of those 12 mics
sender_size = 20

music_dets_paths = [i for i in glob.glob(
    "../processed_data/detections/*") if not "_gt" in i and "music" in i and not "music_0013" in i]
music_gt_paths = [i + "_gt" for i in music_dets_paths]
music_gt_positions_paths = [
    i + "_gt_positions" for i in music_dets_paths]

probs = []
inliers = []
positions = []

for j in range(len(music_dets_paths)):
    dets = pickle.load(open(music_dets_paths[j], "rb"))
    gt = pickle.load(open(music_gt_paths[j], "rb"))
    gt_positions = pickle.load(open(music_gt_positions_paths[j], "rb"))

    # gt = np.load(music_gt_paths[j])
    # gt_positions = np.load(music_gt_positions_paths[j])

    good_index = np.where(np.logical_not(
        np.any(np.isnan(gt_positions["speaker"]), axis=0)))[0]
    n = len(good_index) // sender_size
    indx = np.random.permutation(len(good_index))[
        :(n * sender_size)].reshape((20, n))
    indx = good_index[indx]

    for i in range(n):
        temp = dets[:, :, indx[:, i]][np.tril_indices(dets.shape[0], k=-1)]
        temp2 = gt[:, :, indx[:, i]][np.tril_indices(dets.shape[0], k=-1)]
        gt_positions
        # inlier if within of two dm ~= 60 samples
        inl = np.abs(temp - temp2) < 60

        pos_loc = {"mics": torch.tensor(gt_positions["mics"][:, :, indx[0, i]]), "speaker": torch.tensor(
            gt_positions["speaker"][:, indx[:, i]]).T} # indx[0, i] because mics are stationary

     #   if np.sum(np.isnan(pos_loc["speaker"])) != 0:
    #        print(pos_loc)
   #         raise Exception("found a nan")
        probs.append(torch.tensor(temp))
        inliers.append(torch.tensor(inl))
        positions.append(pos_loc)

problems = np.stack(probs, axis=0)
inliers = np.stack(inliers, axis=0)
positions = np.stack(positions, axis=0)


for i in  range(len(positions)): # Fixing coordinate system so that first microphone is at origin, second is on  x-axis and third is on xy-plane
    # translation
    positions[i]["speaker"] = positions[i]["speaker"] - positions[i]["mics"][0,:]
    positions[i]["mics"] = positions[i]["mics"] - positions[i]["mics"][0,:]
    #rotation
    R = torch.zeros(3,3,dtype=torch.float64)
    R[0,:] = positions[i]["mics"][1,:]/positions[i]["mics"][1,:].norm()
    R[1,:] = positions[i]["mics"][2,:] - R[0,:]*(positions[i]["mics"][2,:]@R[0,:])
    R[1,:] = R[1,:]/R[1,:].norm()
    R[2,:] = torch.cross(R[0,:], R[1,:])
    positions[i]["mics"] = positions[i]["mics"]@R.T
    positions[i]["speaker"] = positions[i]["speaker"]@R.T

if not os.path.exists("../processed_data"):
    os.mkdir("../processed_data")
if not os.path.exists("../processed_data/test_set"):
    os.mkdir("../processed_data/test_set")
pickle.dump(inliers, open("../processed_data/test_set/inliers.pkl", "wb"))
pickle.dump(problems, open("../processed_data/test_set/problems.pkl", "wb"))
pickle.dump(positions, open("../processed_data/test_set/positions.pkl", "wb"))

