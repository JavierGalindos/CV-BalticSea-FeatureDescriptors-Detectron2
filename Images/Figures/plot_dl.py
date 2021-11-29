import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

# set height of bar
# COCO
COCO =  [82.575,99.655]
PASCAL =   [55.680, 85.469]


# pascal voc
# SIFT =  [100*0.0182, 100*0.0372, 100*0.1722]
# ORB =   [100*0.1500, 100*0.0106, 100*0.0389]
# BRIEF = [100*0.0182, 100*0.1383, 100*0]

# Set position of bar on X axis
br1 = np.arange(len(COCO))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, COCO, color ='r', width = barWidth,
        edgecolor ='grey', label ='Training set')
plt.bar(br2, PASCAL, color ='b', width = barWidth,
        edgecolor ='grey', label ='Validation set')


# Adding Xticks
plt.xlabel('Metric', fontweight ='bold', fontsize = 12)
plt.ylabel('mAP', fontweight ='bold', fontsize = 12)
plt.xticks([r + barWidth for r in range(len(COCO))],
        ['COCO', 'Pascal VOC'])
plt.title('mAP in Training and Validation sets', fontweight ='bold')
#plt.title('PASCAL VOC mAP for different classes', fontweight ='bold')

plt.legend()
plt.show()