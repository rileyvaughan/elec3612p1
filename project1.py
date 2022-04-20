import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import svm
from skimage import io
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt

##########################################
###### DATA MODELLING AND SEPARATION #####
##########################################

# The creation of a dictionary handles the mapping of labels to each set of photos, and the image folder structures

classes = ['1a','1b','1c','1d','1e','1f','1g','1h','1i','1j','1k','1l','1m','1n','1o','1p','1q','1r','1s','1t'] 
data = dict.fromkeys(classes,[])
for elem in data:
    data[elem] = io.ImageCollection('./cropped/{}/face/*.pgm'.format(elem))

# Create list structures which will contain all of the data

allImages = []
allLabels = []
min_max_scaler = preprocessing.MinMaxScaler()

# Put all of the data in those empty lists!

for k, v in data.items():
    for elem in v:
        newElem = np.asarray(elem)
        scaled = min_max_scaler.fit_transform(newElem)
        allImages.append(scaled.flatten()) # Make each of the images vectorized and one dimensional
        allLabels.append(classes.index(k)) # Make the labels match up to the images

# Shuffle up all of the data between a training and test set, making sure the training/test split is 80/20

imagesTrain, imagesTest, labelsTrain, labelsTest = train_test_split(allImages, allLabels, train_size=0.8)




####################################
###### SUPPORT VECTOR MODEL ########
####################################

classifSVM = OneVsRestClassifier(estimator=svm.SVC(random_state=0))
modelSVM = classifSVM.fit(imagesTrain, labelsTrain).predict(imagesTest)

reportSVM = classification_report(labelsTest,modelSVM)
#rucSVM = roc_auc_score(labelsTest,modelSVM,multi_class='ovr')

cm = confusion_matrix(labelsTest, modelSVM, labels=classifSVM.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifSVM.classes_)
disp.plot()

print('Report on the performance of the SVM model:\n' + reportSVM + '\n')

####################################
###### LOGISTIC REGRESSION #########
####################################

classifLR = LogisticRegression(solver='saga',random_state=0)
modelLR = classifLR.fit(imagesTrain, labelsTrain).predict(imagesTest)

reportLR = classification_report(labelsTest,modelLR)
#rucSVM = roc_auc_score(labelsTest,modelSVM,multi_class='ovr')

cmLR = confusion_matrix(labelsTest, modelLR, labels=classifLR.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cmLR, display_labels=classifSVM.classes_)
disp1.plot()



print('Report on the performance of the SVM model:\n' + reportLR + '\n')
plt.show()
##########################################
###### PRINCIPAL COMPONENT ANALYSIS ######
##########################################




##########################################
###### LINEAR DISCRIMINANY ANALYSIS ######
##########################################

