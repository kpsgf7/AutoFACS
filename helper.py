import numpy as np
import cv2
def draw_face(face, mesh, img, color=(0,0,255)):

	for i in range(0,len(face), 2):
		x =int(face[i])
		y =int(face[i+1])
		cv2.circle(img, (x,y), 2, color,-1)

	for i in range(0, len(mesh), 2):

		pt1 = mesh[i]
		pt2 = mesh[i+1]

		pt1_x = int(face[pt1*2])
		pt1_y = int(face[pt1*2 +1])

		pt2_x = int(face[(pt2*2)])
		pt2_y = int(face[(pt2*2) +1])
        
		cv2.line(img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, 1, cv2.LINE_AA, 0)

def draw_face_by_num(face, img, color=(0,0,255)):

	for i in range(0,len(face),2):
		x =int(round( face[i]))
		y =int( round(face[i+1]))
		cv2.putText(img,str(i/2),(x,y), cv2.FONT_HERSHEY_SIMPLEX, .2,(255,255,255),1,cv2.LINE_AA)

def draw_face_landmarks(face, img, color=(0,0,255)):
	for i in range(0,len(face),2):
		x =int(round( face[i]) + 245)
		y =int( round(face[i+1]) + 320)
		cv2.circle(img, (x,y), 2, color,-1)

def main():

	landmark_path = "mean_face.txt"

	landmarks=np.empty(136) # ck dataset has 68 landmarks. Will need to change this for other sets.

	mesh_indices = np.empty(666, dtype = int)

	with open(landmark_path, "r") as file:
		# first read out the points for the mean face
		for i in range(0, len(landmarks), 2):
			points = ((file.readline()).strip("\n")).split(",")
			landmarks[i] = float(points[0])
			landmarks[i+1]= float(points[1])


		# then read out the point indices for the mesh
		for i in range(0, len(mesh_indices),2):
			points = ((file.readline()).strip("\n")).split(",")
			mesh_indices[i] = int(float(points[0]))
			mesh_indices[i+1] = int(float(points[1]))


	# redraw everything just to prove it worked
	blank =np.zeros(shape=[490, 640, 3], dtype=np.uint8)

	draw_face(landmarks, mesh_indices, blank, color=(0,255,0))
	#draw_face_by_num(landmarks, blank)

	cv2.imshow("NEUTRAL",blank)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__=="__main__":
	main()