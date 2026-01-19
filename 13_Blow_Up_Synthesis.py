## "Barycenters" are synthesized/interpreted as convex combinations of the blow-up templates
## using the function "blow_up" from "utils"
## We use 3D Data - point clouds
## We consider 2 and 3 templates
## and run the blow-up algorithm of all the templates 
## with respect to (w.r.t.) template 1
## vs. w.r.t template 2 
## vs. w.r.t template 3 (in case we are using 3 templates)
## As a conclusion, the user will notice that convex combinations of blow-up templates
## won't lead to GW barycenters when considering MORE than 2 templates
## (for 2 templates there are theoretical guarantees:
## the convex combinations of their blow-ups determined Geodesics in GW space)
## Finally, when considering 2 templates, we compare the two methods for synthesizing GW barycenters:
## via the pre-defined POT functions (fixed-point iteration) and via convex combination of blow-up templates


import matplotlib.pyplot as plt
import numpy as np
import trimesh
import os
import scipy as sp
from sklearn.manifold import MDS


import ot   # POT: Python Optimal Transport library

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


# Path to the downloaded dataset
dataset_path = utils.load_pointcloud3d()  # The path you got from kagglehub


# Create an MDS instance for visualization
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)


## GET TEMPLATES

# List of different airplane sample files (select 3 or 2)
airplane_files = [
    'airplane_0236.off',    'airplane_0435.off',    #'airplane_0215.off'
]

#number of templates
n_temp = len(airplane_files)

print('Getting 3D point cloud templates and their blow-ups')
# Bounds for sample points from the mesh surface
# (change them to get point cloud templates with the same or different number of sampled point)
l_bound = 50
u_bound = 100

# Store the sampled points for each airplane
sampled_data = []
# list of dissimilarity matrices
matrix_temp_list = []
# list of measures
measure_temp_list = []

# Loop through each airplane file and sample points
for airplane_file in airplane_files:
    # Construct the full path to the .off file
    sample_file_path = os.path.join(dataset_path, 'ModelNet40', 'airplane', 'train', airplane_file)

    # Load the mesh using trimesh
    mesh = trimesh.load_mesh(sample_file_path)

    #Random number of samples
    num_points_to_sample = np.random.randint(l_bound, u_bound)

    # Sample points from the mesh surface
    sampled_points = mesh.sample(num_points_to_sample)

    # Normalize the points to fit within [0, 1]^3
    min_vals = sampled_points.min(axis=0)
    max_vals = sampled_points.max(axis=0)
    normalized_points = (sampled_points - min_vals) / (max_vals - min_vals)

    # Append the normalized points to the list
    sampled_data.append(normalized_points)

    # Dissimilarity matrices
    dist_matrix = sp.spatial.distance.cdist(normalized_points, normalized_points)
    matrix_temp_list.append(dist_matrix)

    # Measure
    p_s = np.ones(num_points_to_sample) / num_points_to_sample
    measure_temp_list.append(p_s)


## Get vector of weights ##########################################################################
#lambdas_list =  np.random.dirichlet(np.ones(n_temp), size=1)[0] # generates random samples from a Dirichlet distribution, which is a common way to generate probability distributions over a simplex.
lambdas_list = np.ones(n_temp)/n_temp      # (uniform)

###################################################################################################
## Templates blow-up w.r.t. the first template
print('Blow-up w.r.t. template 1 (reference)')
B1 = matrix_temp_list[0]
b1 = measure_temp_list[0]
B1, b1, temp_blow_up1 = utils.blow_up(matrix_temp_list, measure_temp_list, B1, b1)
print('Size of the blow-up: ', B1.shape[0])

## Synthesize a GW-Barycenter as convex combination of blow-up templates ##########################
print('Synthesizing a GW-Barycenter as convex combination of blow-up templates (denoted Bary1)')
Bary1 = sum(lambdas_list[j] * temp_blow_up1[j] for j in range(n_temp))
###################################################################################################



## test if we have created a barycenter thru the blow-up method
a = utils.get_lambdas_blowup_matrix(temp_blow_up1, Bary1, b1)
print(f'Barycenter test: {a}') #if the value is zero, we have a barycenter



###################################################################################################
## Templates blow-up w.r.t. the second template
print('Blow-up w.r.t. template 2 (reference)')
B2 = matrix_temp_list[1]
b2 = measure_temp_list[1]
B2, b2, temp_blow_up2 = utils.blow_up(matrix_temp_list, measure_temp_list, B2, b2)
print('Size of the blow-up: ', B2.shape[0])

## Synthesize a GW-Barycenter as convex combination of blow-up templates ##########################
print('Synthesizing a GW-Barycenter as convex combination of blow-up templates (denoted Bary2)')
Bary2 = sum(lambdas_list[j] * temp_blow_up2[j] for j in range(n_temp))
###################################################################################################

## test if we have created a barycenter thru the blow-up method
a = utils.get_lambdas_blowup_matrix(temp_blow_up2, Bary2, b2)
print(f'Barycenter test: {a}') #if the value is zero, we have a barycenter


###################################################################################################
if n_temp == 3:
    ## Templates blow-up w.r.t. the third template
    print('Blow-up w.r.t. template 3 (reference)')
    B3 = matrix_temp_list[2]
    b3 = measure_temp_list[2]
    B3, b3, temp_blow_up3 = utils.blow_up(matrix_temp_list, measure_temp_list, B3, b3)
    print('Size of the blow-up: ', B3.shape[0])

    ## Synthesize a GW-Barycenter as convex combination of blow-up templates ######################
    print('Synthesizing a GW-Barycenter as convex combination of blow-up templates (denoted Bary3)')
    Bary3 = sum(lambdas_list[j] * temp_blow_up3[j] for j in range(n_temp))
###################################################################################################



print('Sanity check:')
## Between different blow-ups for template 1
print('Comparison between different blow-ups for template 1, all of them show be weak isomorphic, thus we should get GW ~ 0')
gromov_distance = ot.gromov.gromov_wasserstein(temp_blow_up2[0], temp_blow_up1[0], b2, b1, log=True)[1]
gw_dist = gromov_distance['gw_dist']
print(f'GW(Temp1,Temp1): {gw_dist}')

if n_temp == 3:
    gromov_distance = ot.gromov.gromov_wasserstein(temp_blow_up2[0], temp_blow_up3[0], b2, b3, log=True)[1]
    gw_dist = gromov_distance['gw_dist']
    print(f'GW(Temp1,Temp1): {gw_dist}')

    gromov_distance = ot.gromov.gromov_wasserstein(temp_blow_up3[0], temp_blow_up1[0], b3, b1, log=True)[1]
    gw_dist = gromov_distance['gw_dist']
    print(f'GW(Temp1,Temp1): {gw_dist}')


## Between different blow-ups for template 2
print('Comparison between different blow-ups for template 2, all of them show be weak isomorphic, thus we should get GW ~ 0')
gromov_distance = ot.gromov.gromov_wasserstein(temp_blow_up2[1], temp_blow_up1[1], b2, b1, log=True)[1]
gw_dist = gromov_distance['gw_dist']
print(f'GW(Temp2,Temp2): {gw_dist}')

if n_temp == 3:
    gromov_distance = ot.gromov.gromov_wasserstein(temp_blow_up2[1], temp_blow_up3[1], b2, b3, log=True)[1]
    gw_dist = gromov_distance['gw_dist']
    print(f'GW(Temp2,Temp2): {gw_dist}')

    gromov_distance = ot.gromov.gromov_wasserstein(temp_blow_up3[1], temp_blow_up1[1], b3, b1, log=True)[1]
    gw_dist = gromov_distance['gw_dist']
    print(f'GW(Temp2,Temp2): {gw_dist}')


## Between different blow-ups for template 3
if n_temp == 3:
    print('Comparison between different blow-ups for template 3, all of them show be weak isomorphic, thus we should get GW ~ 0')
    gromov_distance = ot.gromov.gromov_wasserstein(temp_blow_up2[2], temp_blow_up1[2], b2, b1, log=True)[1]
    gw_dist = gromov_distance['gw_dist']
    print(f'GW(Temp3,Temp3): {gw_dist}')

    gromov_distance = ot.gromov.gromov_wasserstein(temp_blow_up2[2], temp_blow_up3[2], b2, b3, log=True)[1]
    gw_dist = gromov_distance['gw_dist']
    print(f'GW(Temp3,Temp3): {gw_dist}')

    gromov_distance = ot.gromov.gromov_wasserstein(temp_blow_up3[2], temp_blow_up1[2], b3, b1, log=True)[1]
    gw_dist = gromov_distance['gw_dist']
    print(f'GW(Temp3,Temp3): {gw_dist}')

print('End of sanity check.')
###################################################################################################


## COMPARISON #####################################################################################
print('Comparison between the convex combinations with the same vector of coordinates lambda:')

gromov_distance = ot.gromov.gromov_wasserstein(Bary1, Bary2, b1, b2, log=True)[1]
gw_dist12 = gromov_distance['gw_dist']
print(f'GW(Bary1,Bary2): {gw_dist12}')
if gw_dist12 <= 1e-16 :
    print('Conclusion: the different blow_ups are compatible and achieve the same GW barycenter')
else:
    print('Conclusion: the error is big indicating that by using convex combinations of some blow_up templates we do not achieve true GW barycenter as the blow_ups depend on the reference')


if n_temp == 3:
    gromov_distance = ot.gromov.gromov_wasserstein(Bary2, Bary3, b2, b3, log=True)[1]
    gw_dist23 = gromov_distance['gw_dist']
    print(f'GW(Bary2,Bary3): {gw_dist23}')
    if gw_dist23 <= 1e-16 :
        print('Conclusion: the different blow_ups are compatible and achieve the same GW barycenter')
    else:
        print('Conclusion: the error is big indicating that by using convex combinations of some blow_up templates we do not achieve true GW barycenter as the blow_ups depend on the reference')

    gromov_distance = ot.gromov.gromov_wasserstein(Bary3, Bary1, b3, b1, log=True)[1]
    gw_dist31 = gromov_distance['gw_dist']
    print(f'GW(Bary3,Bary1): {gw_dist31}')
    if gw_dist31 <= 1e-16 :
        print('Conclusion: the different blow_ups are compatible and achieve the same GW barycenter')
    else:
        print('Conclusion: the error is big indicating that by using convex combinations of some blow_up templates we do not achive true GW barycenter as the blow_ups depend on the reference')

###################################################################################################

if n_temp == 2:
    print('Comparison with POT synthesis of GW barycenters')
    M = Bary1.shape[0]
    Bary_POT = ot.gromov.gromov_barycenters(
        M, temp_blow_up1, [b1,b1], b1, lambdas_list, max_iter=4500, tol=1e-15)

    gromov_distance = ot.gromov.gromov_wasserstein(Bary1, Bary_POT, b1, b1, log=True)[1]
    gw_dist = gromov_distance['gw_dist']
    print(f'GW(Bary_POT,Bary_ConvexComb): {gw_dist}')



