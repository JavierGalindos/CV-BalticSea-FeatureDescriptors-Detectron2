import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

# set height of bar
# COCO
SIFT =  [100*0.0058, 100*0.0137, 100*0.0746]
ORB =   [100*0.0656, 100*0.0038, 100*0.0151]
BRIEF = [100*0.0045, 100*0.0638, 100*0]

# pascal voc
# SIFT =  [100*0.0182, 100*0.0372, 100*0.1722]
# ORB =   [100*0.1500, 100*0.0106, 100*0.0389]
# BRIEF = [100*0.0182, 100*0.1383, 100*0]

# Set position of bar on X axis
br1 = np.arange(len(SIFT))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, SIFT, color ='r', width = barWidth,
        edgecolor ='grey', label ='SIFT')
plt.bar(br2, ORB, color ='g', width = barWidth,
        edgecolor ='grey', label ='ORB')
plt.bar(br3, BRIEF, color ='b', width = barWidth,
        edgecolor ='grey', label ='BRIEF')

# Adding Xticks
plt.xlabel('Class', fontweight ='bold', fontsize = 12)
plt.ylabel('mAP', fontweight ='bold', fontsize = 12)
plt.xticks([r + barWidth for r in range(len(SIFT))],
        ['Fucus', 'Zostera', 'Furcellaria'])
plt.title('COCO mAP for different classes', fontweight ='bold')
#plt.title('PASCAL VOC mAP for different classes', fontweight ='bold')

plt.legend()
plt.show()