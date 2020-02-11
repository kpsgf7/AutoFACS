# AutoFACS
AutoFACS is a tool to automatically annotate video data using the [Facial Action Coding System.](https://en.wikipedia.org/wiki/Facial_Action_Coding_System) 

The scripts currently in this repo are a proof of concept that closely follows [this paper.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3402717/)

The general process is to take a frame, predict facial landmarks, normalize those landmarks using procrustes analysis, and then feed the distance between landmarks into a Support Vector Machine to detect the presence or absence of an Action Unit. 

Several improvements are currently underway including:
* An improved facial landmark detector that can predict more points with higher accuracy
* Training the Action Unit classifiers on a much larger dataset that was assembled from several different datasets
* Adding a GUI for ease of use

The result of these improvements will be a fully featured GUI based tool for psychology researchers with an improved model for Action Unit prediction. 

