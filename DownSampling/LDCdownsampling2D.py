from paraview.simple import *
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import os

"""Use this code for sampling all 2D LDC Cases both for NS and NSHT"""

#Define Desired Dimensions. Add 1 so that we can compute the Cell Averages
dim_x = 512 + 1
dim_y = 512 + 1

# Path to the input PVTU file
input_file = "sol_00030.pvtu"

# Path to save the output .npz file
output_npz_file = "output_image.npz"

# Load the PVTU file
reader = XMLPartitionedUnstructuredGridReader(FileName=input_file)

# Resample the data to an image
resample = ResampleToImage(Input=reader)
resample.SamplingDimensions = [dim_x, dim_y, 1]  # Set the desired dimensions of the output image

# Update the pipeline to ensure the resampling is done
resample.UpdatePipeline()

# Convert point data to cell data
point_to_cell = PointDatatoCellData(Input=resample)  # Correct filter name
point_to_cell.UpdatePipeline()

# Get the VTK image data from the point_to_cell filter
image_data = point_to_cell.GetClientSideObject().GetOutputDataObject(0)  # Correct method to get the data

# Get the cell data arrays
u_array = image_data.GetCellData().GetArray('u')
if u_array is None:
    raise ValueError("Field 'u' not found in the cell data.")
u_values = vtk_to_numpy(u_array)

v_array = image_data.GetCellData().GetArray('v')
if v_array is None:
    raise ValueError("Field 'v' not found in the cell data.")
v_values = vtk_to_numpy(v_array)

p_array = image_data.GetCellData().GetArray('p')
if p_array is None:
    raise ValueError("Field 'p' not found in the cell data.")
p_values = vtk_to_numpy(p_array)

# Get grid dimensions
dims = image_data.GetDimensions()

# Reshape the cell data arrays to 256 by 256
u_values = u_values.reshape((dims[1]-1, dims[0]-1))
v_values = v_values.reshape((dims[1]-1, dims[0]-1))
p_values = p_values.reshape((dims[1]-1, dims[0]-1))

# Combine u, v, and p arrays into a single 3D array
combined_array = np.stack((u_values, v_values, p_values), axis=-1)  # Shape will be [256, 256, 3]

# Save the combined array as a .npz file
np.savez(output_npz_file, data=combined_array)

print("Combined array shape:", combined_array.shape)
