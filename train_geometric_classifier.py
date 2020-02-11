import numpy as np
import os
import cv2
from neutral_mesh_generator import get_landmarks, procrustes_analysis
from helper import draw_face_landmarks
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

def get_edge_lengths(face, mesh_indices):

	edge_lengths = np.empty(int(len(mesh_indices)/2))

	for i in range(0,len(mesh_indices), 2):
		start = mesh_indices[i]
		end =mesh_indices[i+1]

		edge_lengths[int(i/2)] =  np.sqrt(((face[start*2] - face[end*2])**2) + ((face[start*2 +1] - face[end*2 +1])**2))

	return edge_lengths


def main():

	mean_face_path = "mean_face.txt"
	mean_face=np.empty(136) # ck dataset has 68 landmarks. Will need to change this for other sets.
	mesh_indices = np.empty(666, dtype = int)

	# load the mean face and the indices for the mesh
	with open(mean_face_path, "r") as file:
		# first read out the points for the mean face
		for i in range(0, len(mean_face), 2):
			points = ((file.readline()).strip("\n")).split(",")
			mean_face[i] = float(points[0])
			mean_face[i+1]= float(points[1])

		# then read out the point indices for the mesh
		for i in range(0, len(mesh_indices),2):
			points = ((file.readline()).strip("\n")).split(",")
			mesh_indices[i] = int(float(points[0]))
			mesh_indices[i+1] = int(float(points[1]))

	# dict is to map action units to a position in the label vector
	code_dict = {
		1:1, 2:2, 4:3, 5:4, 6:5,7:6,9:7,10:8,11:9,12:10,
		13:11,14:12,15:13,16:14,17:15,18:16,20:17,21:18,23:19,24:20,
		25:21,26:22,27:23,28:24,29:25,31:26,34:27,38:28,39:29,43:30,
		45:31,22:32,62:33, 64:34, 54:35, 44:36, 30:37, 63:38, 61:39
	}

	subjects_dir = "CK_data_set\\Landmarks\\Landmarks\\"
	all_subjects = sorted(os.listdir(subjects_dir))

	features = np.empty((593, 333)) # each subject has a variable number of sequences so this is hard coded
	labels = np.empty((593, len(code_dict)))

	master_idx = 0
	max_code = 0
	for subj in all_subjects:
		#print(subj)
		for emotion in os.listdir(subjects_dir + subj):
			# just loading the maximum in each emotion for now
			#print(emotion)
			landmark_dir_path = subjects_dir + subj + "\\" + emotion + "\\"

			all_landmarks = sorted(os.listdir(landmark_dir_path))

			neutral = get_landmarks(landmark_dir_path + all_landmarks[0])
			full_emotive = get_landmarks(landmark_dir_path + all_landmarks[-1])

			aligned_neutral = procrustes_analysis(mean_face, neutral)
			aligned_full_emotive = procrustes_analysis(mean_face, full_emotive)

			neutral_edges = get_edge_lengths(aligned_neutral, mesh_indices)
			emotive_edges = get_edge_lengths(aligned_full_emotive, mesh_indices)

			feature_vec = np.array([x-y for x,y in zip(emotive_edges,neutral_edges)])

			features[master_idx] = feature_vec

			facs_path = "CK_data_set\\FACS_labels\\FACS\\" + subj + "\\" + emotion + "\\"
			facs_file = os.listdir(facs_path)[0] # only one file per emotion

			label_vec = np.zeros(len(code_dict))
			with open((facs_path + facs_file), "r") as file:
				for line in file:
					code = int(float(line.strip().split("   ")[0]))
					idx = code_dict[code] - 1
					label_vec[idx] = 1

			labels[master_idx] = label_vec
			master_idx = master_idx+1


	split = int(round(len(features) * .7))

	train_feat = features[:split]
	train_labels = labels[:split]
	test_feat = features[split:]
	test_labels = labels[split:]

	# one svm per au seems to work pretty well. Some don't have enough data to be generalized enough.
	for i in range(0,len(train_labels[0])):
		try:
			clf = SVC(gamma="scale")
			clf.fit(train_feat, train_labels[:,i])
			score = clf.score(test_feat, test_labels[:,i])
			print(str(i) + " " + str(score))
		except:
			print(str(i) + " " + "errored out")



	# labels are formatted as 
	#	AU intensity
	# for example
	# 	9.00000 3.400000
	# means action unit 9 with intensity 3.4

	# the process to generate the feature set will be:
		# load the emotive face
		# load the netural face
		# align both to the mean
		# subtract the neutral face length from the emotive face length for each edge in the mesh
		# the resulting lengths for each edge become the feature vector


	# need to decide which faces to include
		# positive samples are obvious
		# negative samples?

	# need to generate a uniform data set from all the collected data sets


if __name__ =="__main__":
	main()