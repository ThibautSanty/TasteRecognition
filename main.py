# 
# Taste recognition test application
# 	
# A short application to illustrate the difference between two models.
# Here we extract the face from the camera, optimize it for prediction by the two models created
# and then we show the live result on the screen.
#
# # Created by Santy Thibaut

# imports
import cv2
import numpy as np
from keras import models

# variables
ModelClasses = ['bad', 'good']
FacesSequence = []
PredictionOld = "..."
ProgramRunning = True

StyleFont = cv2.FONT_HERSHEY_SIMPLEX
StyleFontScale = 0.7
StyleFontColor = (255,255,255)
StyleLineType = 2
StyleRectangleColor = (0,0,0)
StyleRectangleStroke = 3


global predictionLSTM
global predictionCNN
predictionLSTM = "waiting"
predictionCNN = "waiting"

# functions
def LoadModel(filename=None):
	if (filename != None ):
		return models.load_model("models/" + filename)

	else:
		print("[main] (LoadModel) filename was not given.")
		return None

def GetFaceImage(frame):
	imgGray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
	imgFaceData = ModelFACE.detectMultiScale(imgGray , scaleFactor=2 , minNeighbors=5)

	if (type(imgFaceData) != tuple):
		d = imgFaceData.tolist()
		startX = d[0][0]
		startY = d[0][1]
		endX = d[0][0] + d[0][2]
		endY = d[0][1] + d[0][3]
		face = imgGray[startY:endY , startX:endX]
		return (face , startX , startY , endX , endY)
	else:
		return np.array([]), 0,0,0,0

def PredictReactionWithCNN(face):
	try:
		img = np.reshape( cv2.resize(face,(100,100)) ,[100,100,-1])
		result = ModelCNN.predict( np.reshape(img , [1,100,100,1]) )
		result = result[0].tolist()
		return ModelClasses[result.index(max(result))]

	except Exception as error:
		print("[PredictReactionWithCNN] failed to predict face:\n" + str(error))
		return "failed"

def PredictReactionWithLSTM(faceSequence):
	try:
		seq = np.array(faceSequence)
		result = ModelLSTM.predict( np.reshape(seq,[1,21,100,100,-1]) )
		result = result[0].tolist()
		return ModelClasses[result.index(max(result))]

	except Exception as error:
		print("[PredictReactionWithLSTM] failed to predict face:\n" + str(error))
		return "failed"

# main
try:
	
	ModelCNN = LoadModel("CNN.h5")
	ModelLSTM = LoadModel("lstm2.h5")
	ModelFACE = cv2.CascadeClassifier('models/frontal_face.xml')

	Camera = cv2.VideoCapture(0)

	if (Camera.isOpened()):
		print("[main] camera did open")
		while (ProgramRunning):

			ProgramRunning , frame = Camera.read()
			face , startX , startY , endX , endY = GetFaceImage(frame)

			if (np.size(face)):
				cv2.rectangle(frame , (startX,startY) , (endX, endY) , StyleRectangleColor , StyleRectangleStroke)
				cv2.putText(frame, 'CNN : ' + PredictReactionWithCNN(face) , (int(frame.shape[1]/2),int(frame.shape[0]/1.1)) , StyleFont , StyleFontScale , StyleFontColor , StyleLineType)
				if (len(FacesSequence) == 21):
					PredictionOld = PredictReactionWithLSTM(FacesSequence)
					cv2.putText(frame, 'update' , (20,int(frame.shape[0]/1.2)) , StyleFont , StyleFontScale , StyleFontColor , StyleLineType)
					cv2.putText(frame, 'LSTM : ' + PredictionOld , (20,int(frame.shape[0]/1.1)) , StyleFont , StyleFontScale , StyleFontColor , StyleLineType)
					FacesSequence = []
				else:
					cv2.putText(frame, 'LSTM : ' + PredictionOld , (20,int(frame.shape[0]/1.1)) , StyleFont , StyleFontScale , StyleFontColor , StyleLineType)
					FacesSequence.append(np.reshape( cv2.resize(face,(100,100)) ,[100,100,-1]))

			else:
				cv2.putText(frame, 'CNN : ' + '...' , (int(frame.shape[1]/2),int(frame.shape[0]/1.1)) , StyleFont , StyleFontScale , StyleFontColor , StyleLineType)
				cv2.putText(frame, 'LSTM : ' + PredictionOld , (20,int(frame.shape[0]/1.1)) , StyleFont , StyleFontScale , StyleFontColor , StyleLineType)

			cv2.imshow("taste expressions", frame)

			key = cv2.waitKey(20)
			if key == 27:
				cv2.destroyWindow("taste expressions")
				print("[main] Closed the program")
				break
		
	else:
		print("[main] failed to open the camera")

except Exception as error:
	print("[main] Failed to run script:\n" + str(error))