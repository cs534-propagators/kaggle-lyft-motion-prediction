#>
import gc
import os

import numpy as np

#>
import zarr

import l5kit
from l5kit.data import ChunkedDataset, LocalDataManager

print("l5kit version:", l5kit.__version__)

#>
os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

#>
dm = LocalDataManager()
dataset_path = dm.require('scenes/sample.zarr')
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)

#>
print(zarr_dataset.agents)
print(zarr_dataset.agents.shape)
n = zarr_dataset.agents.shape

#>
def getAgentsChunked(dataset, subsetPercent=1, chunks=10):

    datasetLength = round(len(dataset) * subsetPercent)
    chunkSize = round(datasetLength / chunks)
    print("datasetLength", datasetLength)
    print("chunkSize", chunkSize)
    agents = []
    track_id1_indexes = []
    for i in range(0, datasetLength, chunkSize):

        agentsSubset = dataset[i:i+chunkSize]
        for j in range(0,len(agentsSubset)):

            agent = agentsSubset[j]
            track_id = agent[4]
            if track_id == 1:
                track_id1_indexes.append(i+j)

    return agents, track_id1_indexes

#>
subsetPercent = 1*10**-2
print(subsetPercent)
agents, track_id1_indexes = getAgentsChunked(zarr_dataset.agents, subsetPercent, 100)

#>
dataset_path_test = dm.require('scenes/test.zarr')
zarr_dataset_test = ChunkedDataset(dataset_path_test)
zarr_dataset_test.open()
print(zarr_dataset_test)

#>
test_mask = np.load('../input/lyft-motion-prediction-autonomous-vehicles/scenes/mask.npz')
for k in test_mask.files:
    print("key:",k)
test_mask = test_mask["arr_0"]
print("test_mask", test_mask)
print("test_mask.shape", test_mask.shape)
print("test_mask[0]", test_mask[0])
