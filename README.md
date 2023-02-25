# ML-for-EEG-Data
-> In depth article: https://medium.com/@serjankaur/uncovering-the-surprise-response-using-machine-learning-to-analyze-eeg-data-a2e1afd6d450
In this experiment, a subject is seated in front of a screen and is presented with a series of images. The images include checkerboard patterns that are displayed in either the left or right visual field, and are accompanied by tones that are presented to either the left or right ear. The interval between the stimuli (i.e. the checkerboard patterns and tones) is 750 milliseconds (ms).

Occasionally, a smiley face is presented at the center of the visual field. When this occurs, the subject is asked to press a key with their right index finger as quickly as possible. This task is likely being used to measure the subject’s reaction time to the appearance of the smiley face.

![image](https://user-images.githubusercontent.com/94884804/221376814-441caaff-6964-4a84-a146-e91509ad6c26.png)

The idea is to use only the EEG data feed provided, to detect when the subject is shown the Smiley Face image, before they press the key.
Overall, this experiment is designed to study the subject’s visual and auditory processing, as well as their reaction time. By presenting stimuli to different parts of the visual field and to different ears, the experimenters can study how the subject’s brain processes and integrates this information. The task of pressing a key in response to the smiley face allows the experimenters to measure the subject’s reaction time, which can provide insights into their cognitive processing speed.

The purpose of this experiment is to find a way to detect surprise or anticipation by looking at the electrical activity of the brain. The experimenters hope that by creating a system to detect the P300 component, they will be able to solve the problem of detecting these emotions.

