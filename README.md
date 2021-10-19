# its8030-2021-hw2

Project developed by Omar El Nahhas & Javier Galindos


## Task description
In this task you are asked to solve the problem of detecting underwater plants. Your solution should
be Python 3 code (including type annotations of functions) (F# also accepted) in the root directory of
a repository called its8030-2021-hw2 to be uploaded to Gitlab.cs.ttu.ee. The work can be done in
pairs. More members in a team require a written consent from the lecturer.

Detection of vegetation in the sea is an important tool for the detection of the health of the water
body: http://stateofthebalticsea.helcom.fi/biodiversity-and-its-status/benthic-habitats/

The current task is about automated detection of benthic species of plants in the Baltic Sea:
- Charophyte
- Fucus
- Furcellaria lubricalis
- Mytilus
- Zostera marina

Please note that absence of any vegetation should be as a separate class.
### Task 1: Preprocessing
Collect a set of images suitable for the tasks below of at least 3 species. Write code to preprocess the
images of plants into a uniform size of your choice, e.g. 1024x1024 pixels.
### Task 2: Etalons
Select a set of etalons (e.g. small images containing a sample of some distinctive features) from the
an image to be used for matching similar objects. Aim for a set that would be representative on at
least 50% of the images of the particular species. Think how to deal with rotations.
### Task 3: Baseline
Setting a baseline: Use at least 3 different existing conventional feature detectors provided by
OpenCV to find matches of the etalons in the image. NB! Take into account overlaps and subtract the
appropriate numbers from total scores.

Evaluate on two different images (called task3a.tiff and task3b.tiff) how well the approach works and
which feature detector performs best.
### Task 4:
Improve the baseline by applying deep learning. Key words: OpenCV 4, OpenVINO, ONNX,
Tensorflow, PyTorch. The result needs to be documented, i.e. you should present quantitative results
in a report where you show if and how much the deep learning approach improved your baseline.
Make sure to use appropriate criteria for the measurement!

NB! Aim at using a pretrained network as a basis and apply the concept of transfer learning to adjust
the net for your task.

NB! The work needs to be documented, i.e. you need to include a report where you have
quantitative results of your baseline and improvement in addition to the description of the approach
taken.

To store images in the Git repository you should use Git LFS support.
You may use the AI-Lab environment for GPU accelerated computations.
### Helpful hints:
Labeling tool: CVAT
