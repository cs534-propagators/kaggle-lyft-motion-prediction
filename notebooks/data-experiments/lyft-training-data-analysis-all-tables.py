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
import time
from datetime import datetime

#>
# helper to convert a timedelta to a string (dropping milliseconds)
def deltaToString(delta):
    timeObj = time.gmtime(delta.total_seconds())
    return time.strftime('%H:%M:%S', timeObj)

class ProgressBar:
    
    # constructor
    #   maxIterations: maximum number of iterations
    def __init__(self, maxIterations):
        self.maxIterations = maxIterations
        self.granularity = 100 # 1 whole percent
    
    # start the timer
    def start(self):
        self.start = datetime.now()
    
    # check the progress of the current iteration
    #   # currentIteration: the current iteration we are on
    def check(self, currentIteration, chunked=False):
        if currentIteration % round(self.maxIterations / self.granularity) == 0 or chunked:
            
            percentage = round(currentIteration / (self.maxIterations - self.maxIterations / self.granularity) * 100)
            
            current = datetime.now()
            
            # time calculations
            timeElapsed = (current - self.start)
            timePerStep = timeElapsed / (currentIteration + 1)
            totalEstimatedTime = timePerStep * self.maxIterations
            timeRemaining = totalEstimatedTime - timeElapsed
            
            # string formatting
            percentageStr = "{:>3}%  ".format(percentage)
            remainingStr = "Remaining: {}  ".format(deltaToString(timeRemaining))
            elapsedStr = "Elapsed: {}  ".format(deltaToString(timeElapsed))
            totalStr = "Total: {}\r".format(deltaToString(totalEstimatedTime))
            
            print(percentageStr + remainingStr + elapsedStr + totalStr, end="")

    def end(self):
        print()

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
subsetPercent = 1 #1*10**-1
print("subsetPercent", subsetPercent)
scenesLen = round(len(zarr_dataset.scenes) * subsetPercent)
print("scenesLen",scenesLen)

framesIntervalIndex = 0
agentsIntervalIndex = 1

pb0 = ProgressBar(scenesLen)
pb0.start()

totalDataCount = 0
totalAgentsCount = 0
# TODO: could possibly be faster if we get a bigger subset than currently needed and cache it for when we actually do need it

trainingAgentsDict = {}

scenesSubsetDataset = zarr_dataset.scenes[0:scenesLen]
for sceneIndex in range(0, scenesLen):
    pb0.check(sceneIndex)
    scene = scenesSubsetDataset[sceneIndex]

    framesInterval = scene[framesIntervalIndex]

    frameStart = framesInterval[0]
    frameEnd = framesInterval[1]
    
    framesSubsetDataset = zarr_dataset.frames[frameStart:frameEnd]
    for frameIndex in range(0, len(framesSubsetDataset)):
        frame = framesSubsetDataset[frameIndex]
        
        agentsInterval = frame[agentsIntervalIndex]
        
        agentStart = agentsInterval[0]
        agentEnd = agentsInterval[1]
        
        agentsSubsetDataset = zarr_dataset.agents[agentStart:agentEnd]
        for agentIndex in range(0, len(agentsSubsetDataset)):
            agent = agentsSubsetDataset[agentIndex]
            
            centroid = agent[0]
            yaw = agent[2]
            velocity = agent[3]
            track_id = agent[4]

            data = []
            data.append(centroid[0])
            data.append(centroid[1])
            data.append(yaw)
            data.append(velocity[0])
            data.append(velocity[1])
            
            if track_id not in trainingAgentsDict:
                trainingAgentsDict[track_id] = {}
            
            trainingAgentDict = trainingAgentsDict[track_id]
            
            if sceneIndex not in trainingAgentDict:
                trainingAgentDict[sceneIndex] = []
                
            trainingAgentScene = trainingAgentDict[sceneIndex]
            
            trainingAgentScene.append(data)
            totalDataCount += 1
        totalAgentsCount += 1

#>
print("totalDataCount",totalDataCount)
print("totalAgentsCount",totalAgentsCount)

#>
print(len(trainingAgentsDict))
print(len(trainingAgentsDict[1]))
print(len(trainingAgentsDict[1][0]))

#>

