#!/usr/bin/env python
# coding: utf-8

# In[]:


import gc
import os
from pathlib import Path
import random
import sys

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import scipy as sp


import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.templates.default = "plotly_dark"

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- setup ---
pd.set_option('max_columns', 50)


# In[]:


import zarr

import l5kit
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable

from matplotlib import animation, rc
from IPython.display import HTML

rc('animation', html='jshtml')
print("l5kit version:", l5kit.__version__)


# In[]:


os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"


# In[]:


dm = LocalDataManager()
dataset_path = dm.require('scenes/sample.zarr')
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)


# ### scenes
# 
# ```
# SCENE_DTYPE = [
#     ("frame_index_interval", np.int64, (2,)),
#     ("host", "<U16"),  # Unicode string up to 16 chars
#     ("start_time", np.int64),
#     ("end_time", np.int64),
# ]
# ```

# In[]:


print("scenes", zarr_dataset.scenes, "\n")

scene = zarr_dataset.scenes[0] # single frame
print("scene", scene, "\n")

print("len(scenes)", len(zarr_dataset.scenes))
print("len(scene)", len(scene), "\n")

print("frame_index_interval", scene[0])
print("host", scene[1])
print("start_time", scene[2])
print("end_time", scene[3])
print("end-start", scene[3]-scene[2], "\n")

scene = zarr_dataset.scenes[1] # single frame
print("frame_index_interval", scene[0])
print("host", scene[1])
print("start_time", scene[2])
print("end_time", scene[3])
print("end-start", scene[3]-scene[2], "\n")

scene = zarr_dataset.scenes[2] # single frame
print("frame_index_interval", scene[0])
print("host", scene[1])
print("start_time", scene[2])
print("end_time", scene[3])
print("end-start", scene[3]-scene[2], "\n")

for i in range(0,3):
    scene = zarr_dataset.scenes[i]
    frame_index_interval = scene[0]
    print("frame_index_interval {}".format(i), frame_index_interval)


# ### frames
# 
# ```
# FRAME_DTYPE = [
#     ("timestamp", np.int64),
#     ("agent_index_interval", np.int64, (2,)),
#     ("traffic_light_faces_index_interval", np.int64, (2,)),
#     ("ego_translation", np.float64, (3,)),
#     ("ego_rotation", np.float64, (3, 3)),
# ]
# ```
# 

# In[]:


print("frames", zarr_dataset.frames, "\n")

frame = zarr_dataset.frames[0] # single frame
print("frame", frame, "\n")

print("len(frames)", len(zarr_dataset.frames))
print("len(frame)", len(frame), "\n")

for i in range(0,3):
    frame = zarr_dataset.frames[i]
    ego_translation = frame[3]
    print("ego_translation {}".format(i), ego_translation)


# ### agents
# 
# ```
# AGENT_DTYPE = [
#     ("centroid", np.float64, (2,)),
#     ("extent", np.float32, (3,)),
#     ("yaw", np.float32),
#     ("velocity", np.float32, (2,)),
#     ("track_id", np.uint64),
#     ("label_probabilities", np.float32, (len(LABELS),)),
# ]
# ```
# 

# In[]:


print("agents", zarr_dataset.agents, "\n")

agent = zarr_dataset.agents[0]
print("agent", agent, "\n")

print("len(agents)", len(zarr_dataset.agents))
print("len(agent)", len(agent), "\n")

centroid = agent[0]
extent =   agent[1]
yaw =      agent[2]
print("centroid", centroid)
print("extent", extent)
print("yaw", yaw)


# In[]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np


# In[]:


x = []
timestamp = []
for i in range(0, len(zarr_dataset.frames)):
    frame = zarr_dataset.frames[i]
    x.append(i)
    timestamp.append(frame[0])
plt.plot(x, timestamp, 'o', color='black')


# In[]:


x = []
timestamp = []
for i in range(0, 746):
    frame = zarr_dataset.frames[i]
    x.append(i)
    timestamp.append(frame[0])
plt.plot(x, timestamp, 'o', color='black')


# In[]:


x = []
timestamp = []
for i in range(248-5, 248+5):
    frame = zarr_dataset.frames[i]
    x.append(i)
    timestamp.append(frame[0])
plt.plot(x, timestamp, 'o', color='black')


# In[]:


x = []
timestamp = []
for i in range(497-5, 497+5):
    frame = zarr_dataset.frames[i]
    x.append(i)
    timestamp.append(frame[0])
plt.plot(x, timestamp, 'o', color='black')


# In[]:


#x = []
#timestamp = []
for i in range(0, 10):#len(zarr_dataset.frames)):
    frame = zarr_dataset.frames[i]
    #x.append(i)
    #timestamp.append(frame[1])
    agent_index_interval = frame[1]
    print(agent_index_interval, agent_index_interval[1]-agent_index_interval[0])
#plt.plot(x, timestamp, 'o', color='black')


# In[]:


ego_translation_x = []
ego_translation_y = []
for i in range(0, len(zarr_dataset.frames)):
    frame = zarr_dataset.frames[i]
    ego_translation_x.append(frame[3][0])
    ego_translation_y.append(frame[3][1])
plt.plot(ego_translation_x, ego_translation_y, 'o', color='black')


# In[]:


centroid_x = []
centroid_y = []
for i in range(0, 1000):#len(zarr_dataset.agents)):
    agent = zarr_dataset.agents[i]
    centroid = agent[0]
    centroid_x.append(centroid[0])
    centroid_y.append(centroid[1])
plt.plot(centroid_x, centroid_y, 'o', color='black')


# In[]:


centroid_x = []
centroid_y = []
for i in range(1000, 2000):#len(zarr_dataset.agents)):
    agent = zarr_dataset.agents[i]
    centroid = agent[0]
    centroid_x.append(centroid[0])
    centroid_y.append(centroid[1])
plt.plot(centroid_x, centroid_y, 'o', color='black')


# In[]:


#centroid_x = []
#centroid_y = []
for i in range(0, 1000):#len(zarr_dataset.agents)):
    agent = zarr_dataset.agents[i]
    track_id = agent[4]
    print(track_id)
    #centroid = agent[0]
    #centroid_x.append(centroid[0])
    #centroid_y.append(centroid[1])
#plt.plot(centroid_x, centroid_y, 'o', color='black')


# In[]:



centroid_x = []
centroid_y = []

maxAgents = 1000
for i in range(0, maxAgents):
    centroid_x.append([])
    centroid_y.append([])
    
for i in range(0, 10000):#len(zarr_dataset.agents)):
    agent = zarr_dataset.agents[i]
    track_id = agent[4]
    if track_id < maxAgents + 1:
        #print(i)
        centroid = agent[0]
        centroid_x[int(track_id)-1].append(centroid[0])
        centroid_y[int(track_id)-1].append(centroid[1])

            
r = lambda: random.randint(0,255)
for i in range(0, maxAgents):
    plt.plot(centroid_x[i], centroid_y[i], 'o', color='#%02X%02X%02X' % (r(),r(),r()))


# In[]:


print(len(centroid_x))

print(len(centroid_x[0]))
print(len(centroid_x[1]))
print(len(centroid_x[2]))


# ### traffic_light_faces
# 
# ```
# TL_FACE_DTYPE = [
#     ("face_id", "<U16"),
#     ("traffic_light_id", "<U16"),
#     ("traffic_light_face_status", np.float32, (len(TL_FACE_LABELS,))),
# ]
# ```

# In[]:


#zarr_dataset.traffic_light_faces

print("tl_faces", zarr_dataset.tl_faces, "\n")

tl_face = zarr_dataset.tl_faces[0] # single traffic_light_face
print("tl_face", tl_face, "\n")

print("len(tl_faces)", len(zarr_dataset.tl_faces))
print("len(tl_face)", len(tl_face), "\n")

tl_id = tl_face[1]
tl_face_status = tl_face[2]

print(tl_face[0])
print(tl_id)
print(tl_face_status) # [active, inactive, unknown]

for i in range(0, 1000):
    tl_face = zarr_dataset.tl_faces[i] # single traffic_light_face
    tl_face_status = tl_face[2]
    if tl_face_status[0] > 0.2:
        print(tl_face)
        


# In[]:


status = [0,0,0]
for i in range(0,len(zarr_dataset.tl_faces)):
    tl_face = zarr_dataset.tl_faces[i] # single traffic_light_face
    tl_face_status = tl_face[2]
    status = status + tl_face_status
    
    if i % int(len(zarr_dataset.tl_faces) / 100) == 0:
        print(str(round(i / len(zarr_dataset.tl_faces) * 100)) + "%", status)

print(status)
        


# In[]:


dataset_path = dm.require('scenes/test.zarr')
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)


# In[]:




