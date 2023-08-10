import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import functools
import sys

sys.path.append('../src')
import tdoa_datasets_module as tdoa

# Settings
# sizes of windows to run detections methods on
# runs methods on all data with ground truth
experiments, recordings = tdoa.get_data_paths(
    "../data/", with_ground_truth=True)
results_path = "../processed_data/detections"

window_size = 10000

# create folders
if not os.path.exists(results_path):
    os.mkdir(results_path)

for experiment in experiments:
    for recording_folder in recordings[experiments[0]]:
        tdoa_chunk_estimation, tdoa_chunk_gt = tdoa.evaluate_tdoa_estimator_on_recording(
            tdoa.gcc_phat, recording_folder, chunk_length=window_size)
        np.save(results_path + "/" + recording_folder.split("/")[-1], tdoa_chunk_estimation)
        np.save(results_path + "/" + recording_folder.split("/")[-1] +  "_gt", tdoa_chunk_gt)
