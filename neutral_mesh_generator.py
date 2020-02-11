import cv2
import os
import numpy as np
from scipy.linalg import norm
from math import sin,cos,atan


def get_translation(shape):

	mean_x = np.mean(shape[::2]).astype(np.int)
	mean_y = np.mean(shape[1::2]).astype(np.int)

	return np.array([mean_x, mean_y])

def translate(shape):

	mean_x, mean_y = get_translation(shape)
	shape[::2] -= mean_x
	shape[1::2] -= mean_y

def get_rotation_scale(reference_shape, shape):

	a = np.dot(shape, reference_shape) / norm(reference_shape)**2

	#separate x and y for the sake of convenience
	ref_x = reference_shape[::2]
	ref_y = reference_shape[1::2]

	x = shape[::2]
	y = shape[1::2]

	b = np.sum(x*ref_y - ref_x*y) / norm(reference_shape)**2

	scale = np.sqrt(a**2+b**2)
	theta = atan(b / max(a, 10**-10)) #avoid dividing by 0

	return round(scale,1), round(theta,2)

def get_rotation_matrix(theta):
	return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def rotate(shape, theta):

	matr = get_rotation_matrix(theta)

	#reshape so that dot product is eascily computed
	temp_shape = shape.reshape((-1,2)).T

	#rotate
	rotated_shape = np.dot(matr, temp_shape)

	return rotated_shape.T.reshape(-1)

def procrustes_analysis(reference_face, face):
	#copy both shapes in caseoriginals are needed later.
    temp_ref = np.copy(reference_face)
    temp_sh = np.copy(face)
 
    translate(temp_ref)
    translate(temp_sh)
    
    #get scale and rotation
    scale, theta = get_rotation_scale(temp_ref, temp_sh)
    
    #scale, rotate both shapes
    temp_sh = temp_sh / scale
    aligned_face = rotate(temp_sh, theta)
    
    return aligned_face

def procrustes_distance(reference_shape, shape):

	ref_x = reference_shape[::2]
	ref_y = reference_shape[1::2]

	x = shape[::2]
	y = shape[1::2]

	dist = np.sum(np.sqrt((ref_x - x)**2 + (ref_y - y)**2))

	return dist

def generalized_procrustes(faces):

	# initialize mean face as first face
	mean_face = faces[0]

	new_faces = np.zeros(faces.shape)

	current_distance = 0
	while True:

		new_faces[0] = mean_face

		for idx in range(1,len(faces)):
			new_faces[idx] = procrustes_analysis(mean_face, faces[idx])

		new_mean = np.mean(new_faces, axis=0)
		new_distance = procrustes_distance(new_mean, mean_face)

		if new_distance == current_distance:
			break

		#align the new_mean to old mean
		new_mean = procrustes_analysis(mean_face, new_mean)

		#update mean and distance
		mean_face = new_mean
		current_distance = new_distance

	return mean_face


def get_landmarks(landmark_path):

	landmarks=np.empty(136) # ck dataset has 68 landmarks. Will need to change this for other sets.

	with open(landmark_path, "r") as file:
		idx = 0
		for line in file:
			points = (line.strip()).split("   ")
			landmarks[idx] = float(points[0])
			landmarks[idx+1]= float(points[1])
			idx = idx +2

	return landmarks

def main():

	num_subjs = 50

	subjects_dir = "CK_data_set\\extended-cohn-kanade-images\\cohn-kanade-images\\"
	training_subjects = sorted(os.listdir(subjects_dir))[:num_subjs]

	# load the faces
	all_faces = np.empty((num_subjs, 136)) # ck dataset has 68 landmarks. Will need to change this for other sets.
	for idx in range(0,num_subjs):
		load_path = "CK_data_set\\Landmarks\\Landmarks\\" + training_subjects[idx] + "\\001\\" + training_subjects[idx] +"_001_00000001_landmarks.txt"
		all_faces[idx] = get_landmarks(load_path)

	# compute the generalized face
	mean_face = generalized_procrustes(all_faces)

	# write the points out and then write the edges out
	with open("mean_face.txt", "w") as handle:
		subdiv = cv2.Subdiv2D((0,0,490,640))
		mean_face_dict = {} # dict of point values to numbers i.e mean_face_dict[(x,y)] = point_idx

		for i in range(0,len(mean_face), 2):
			x = int(mean_face[i] + 245) # need to consider how to fix this adjusment
			y = int(mean_face[i+1] + 320)
			handle.write(str(x) + "," + str(y) + "\n")

			mean_face_dict[(x,y)] = i/2

			subdiv.insert((x,y))

		triangles = subdiv.getTriangleList() # have to use this because the other methods are poorly documented and dont seem to work

		edges = {}

		for t in triangles:
			pt1 = (t[0], t[1])
			pt2 = (t[2], t[3])
			pt3 = (t[4], t[5])


			# we have a dict with the points as keys and the number as values
			# need to use that to create a dict that has end points as keys and point numbers as values. 
			# then write out the point numbers
			# garuntees uniqueness in the lines

			lines = [pt1+pt2, pt2+pt3, pt3+pt1]

			for line in lines:
				if line not in edges:
					edges[line] = [mean_face_dict[line[:2]], mean_face_dict[line[2:]]]

		edges_out = [str(x[0]) + "," + str(x[1]) + "\n" for x in edges.values()]
		print(len(edges_out))
		handle.writelines(edges_out)

		
if __name__=="__main__":
	main()