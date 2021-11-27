import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import re
from feat_matchers import orb_detector, brief_detector, sift_detector, fill_bounding_boxes

from skimage import data, filters, color, morphology
from skimage.segmentation import flood, flood_fill
from skimage.measure import label, regionprops
from bs4 import BeautifulSoup
from mean_average_precision import MetricBuilder

# Read the query image as query_img
# and train image This query image
# is what you need to find in train image
# Save it in the same directory
# with the name image.jpg 


# PICK AN IMAGE 

def fd_detector(method = 'SIFT', lowe_ratio = 0.7, good_match_thresh = 0):
    
    overall_pascal = []
    overall_coco = []
    
    
    process_img_path = './fucus_zoster_val/'
    
    #input argument to choose detector
    for type_of_class in os.listdir('./fucus_zoster_val/'):
        print(type_of_class)
        full_dir_path = os.path.join(process_img_path, type_of_class)
        for img_path in os.listdir(full_dir_path):
            if img_path.endswith('.jpg') or img_path.endswith('.png'):
                full_img_path = os.path.join(full_dir_path, img_path)
                print('Now processing: %s' % img_path)
                train_img_bw = cv2.imread(full_img_path,0)

                grid_per_side = 4
                step = int(train_img_bw.shape[0]/grid_per_side) # 4x4 matrix

                grid_class = np.empty(shape = (grid_per_side, grid_per_side), dtype='object')

                classes = ['Fuc', 'Fur', 'Zos', 'None']

                for x_coord in range(0, train_img_bw.shape[0], step):
                    begin_x = x_coord
                    end_x = x_coord + step
                    for y_coord in range(0, train_img_bw.shape[1], step):
                        begin_y = y_coord
                        end_y = y_coord + step
                        
                        #step through every grid cell
                        img_patch = train_img_bw[begin_x:end_x, begin_y:end_y]
                        
                        class_for_patch = []
                        #now check this image patch with all the etalons, return the max class (or nothing) per img patch
                        for folder in os.listdir('./Etalons'):
                            #get to a particular class of etalon
                            etalon_dir_path = os.path.join('./Etalons', folder)
                            
                            max_goodies = 0
                            
                            for etalon_img_path in os.listdir(etalon_dir_path):
                                
                                etalon_fullpath = os.path.join(etalon_dir_path, etalon_img_path)
                                
                                # Now detect the keypoints and compute
                                # the descriptors for the query image
                                # and train image
                                query_img_bw = cv2.imread(etalon_fullpath,0)
                                #print(etalon_fullpath)
                                if method == 'ORB':
                                    good_matches = orb_detector(query_img_bw, img_patch, lowe_ratio)
                                elif method == 'BRIEF':
                                    good_matches = brief_detector(query_img_bw, img_patch, lowe_ratio)
                                else: #SIFT, best one
                                    good_matches = sift_detector(query_img_bw, img_patch, lowe_ratio)
                                
                                # img3 = cv2.drawMatchesKnn(query_img_bw,queryKeypoints,img_patch,trainKeypoints,good, None, flags=2)
                                max_goodies += good_matches

                            class_for_patch.append(max_goodies)
                        #print((class_for_patch))
                        if np.max(class_for_patch) > good_match_thresh:
                            #deal with multiple maxima at one point
                            m = [i for i,j in enumerate(class_for_patch) if j==np.max(class_for_patch)]
                        
                            pred_patch_class = ''
                            for i in m:
                                pred_patch_class += classes[i]

                        else:
                            pred_patch_class = 'None'
                            
                        #print(pred_patch_class)
                        grid_class[int(begin_x/step),int(begin_y/step)] = pred_patch_class
                
                
                ## second cell
                
                len_per_class = 3
                grid_vect = np.zeros(16, dtype='object')
                nr = 0
                numerical_grid = np.zeros((grid_per_side, grid_per_side), dtype='float')
                for i in range(grid_per_side):
                    for j in range(grid_per_side):
                        #numerical_grid[i][j] = round(len_per_class/len(grid_class[i][j]),2)
                        numerical_grid[i][j] = nr
                        grid_vect[nr] = grid_class[i][j]
                        nr +=1
                
                ## third cell
                classes = ['Fuc', 'Fur', 'Zos', 'None']

                box10 = [0,1,4,5]
                box11 = [1,2,5,6]
                box12 = [2,3,6,7]

                box20 = [4,5,8,9]
                box21 = [5,6,9,10]
                box22 = [6,7,10,11]

                box30 = [8,9,12,13]
                box31 = [9,10,13,14]
                box32 = [10,11,14,15]

                all_windows = [box10, box11, box12, box20, box21, box22, box30, box31,box32]
                windows_names = ['box10', 'box11', 'box12', 'box20', 'box21', 'box22', 'box30', 'box31','box32']
                i = 0
                best_class = np.zeros(9, dtype='object')
                best_class_num = np.zeros(9, dtype='int')
                best_class_conf = np.zeros(9, dtype='float')
                
                #print(grid_vect)
                
                for box in all_windows:

                    fuc = 0
                    fur = 0
                    zos = 0
                    no = 0
                    #print(grid_vect[box])
                    for element in grid_vect[box]:
                        #print(element)
                        #if the are multiple elements, we give it a lower score per class
                        #because we don't award confusion :)
                        if(len(element) == 9):
                            multiplier = (1/3)
                        elif(len(element) == 6):
                            multiplier = (1/2)
                        else:
                            multiplier = 1
                        fuc += multiplier*len(re.findall('Fuc', element))
                        fur += multiplier*len(re.findall('Fur', element))
                        zos += multiplier*len(re.findall('Zos', element))
                        no += multiplier*len(re.findall('None', element))
                    

                    # print(fuc)
                    # print(fur)
                    # print(zos)
                    # print(no)
                    #find all the occurences of maximum class
                    m = [skd for skd,j in enumerate((fuc, fur, zos, no)) if j==np.max((fuc, fur, zos, no))]
                    #more than 2 with same likelihood? then just None to avoid confusion
                    #print(m)
                    if(len(m) > 1):
                        best_class[i] = classes[3] #none
                        best_class_num[i] = 3
                    else:
                        best_class[i] = classes[np.argmax((fuc, fur, zos, no))]
                        best_class_num[i] = np.argmax((fuc, fur, zos, no))
                    
                    best_class_conf[i] = np.round((np.max((fuc, fur, zos, no))/4), 4)
                    
                    #print('%s: %s' %(windows_names[i], best_class[i]))
                    i+=1
                
                ## fourth cell
                best_class = best_class.reshape((3,3))
                best_class_num = best_class_num.reshape((3,3))
                best_class_conf = best_class_conf.reshape((3,3))
                
                #fifth cell
                
                all_indeces = [(0,0), (0,1), (0,2), (1, 0), (1, 1), (1, 2), (2, 0), (2,1), (2, 2)]
                coordinates = np.zeros((3,3), dtype = 'int')

                for index in all_indeces:
                    #get the value of the index
                    x = index[0]
                    y = index[1]
                    if best_class_num[x][y] >= 0:
                        #map to specific class, -1 is fucus, -2 is furcellaria, -3 is Zos and otherwise its None
                        value = (-1 if best_class_num[x][y] == 0 else -2 if best_class_num[x][y] == 1 else -3 if best_class_num[x][y] == 2 else -4)
                        #filled_checkers = flood_fill(best_class_num, index, value)
                        findconn = flood(best_class_num, index)
                    
                        
                        if(np.sum(findconn)) > 1:
                            where_is_it = np.where(findconn == True)
                            for i in range(len(where_is_it[0])):
                                o = where_is_it[0][i]
                                u = where_is_it[1][i]
                                coordinates[o][u] = value
                            
                seperation = coordinates.copy()
                classes, coords = fill_bounding_boxes(coordinates)
                
                
                conf = []

                for i in range(len(classes)):
                    slice = coords[i]
                    this_class = classes[i]
                    find_rectangle = (seperation[slice] == this_class)
                    cropped_conf = best_class_conf[slice]
                    
                    #print(this_class)
                    #print(seperation[slice])
                        
                    conf.append(sum(cropped_conf[find_rectangle])/(np.prod(find_rectangle.shape))) # avg confidence
                
                step = int(256*(4/3))
                pred = []

                for x in range(len(conf)):
                    this_slice = coords[x]
                    x_begin = this_slice[0].start
                    x_end = this_slice[0].stop
                    y_begin = this_slice[1].start
                    y_end = this_slice[1].stop
                    
                    #print(this_slice[0])
                    
                    this_conf = conf[x]
                    this_class = classes[x]
                    
                    class_name = ['Fucus', 'Furcellaria', 'Zostera', 'None']
                    #print(this_class)
                    class_single = class_name[(-this_class-1)] #name string
                    
                    pred.append((x_begin*step, y_begin*step, x_end*step, y_end*step, int(-this_class-1), this_conf))
                    
                    # Plot image
                    
                    color_list = [[0,0,255], [0,255,0], [255,0,0], [0,0,0]]
                    color = color_list[(-this_class-1)]
                    thickness = 2

                    outputstr = class_single + ': ' + str(this_conf)
                            
                    image = cv2.rectangle(cv2.cvtColor(train_img_bw, cv2.COLOR_GRAY2RGB), (int(x_begin*step), int(y_begin*step)), (int(x_end*step + step), int(y_end*step+step)), color, thickness)     
                    image = cv2.putText(image, outputstr, ((int(x_begin*step)+5), int(y_begin*step)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                pred = np.array(pred) 
                #imS = cv2.resize(image, (960, 540))       
                cv2.imwrite(os.path.join('./output_fd/', img_path),image)
                #cv2.moveWindow('FD output', 40,30)  # Move it to (40,30)
                
                
                                    # Reading the data inside the xml
                # file to a variable under the name
                # data
                with open('./all_classes_cv/annotations_valid.xml', 'r') as f:
                    data = f.read()
                
                # Passing the stored data inside
                # the beautifulsoup parser, storing
                # the returned object
                Bs_data = BeautifulSoup(data, "xml")
                
                # Using find() to extract attributes
                # of the first instance of the tag
                b_name = Bs_data.find('image',{'name': img_path})

                box_data = b_name.find('box')
                xbr = int(float(box_data.get('xbr')))
                xtl = int(float(box_data.get('xtl')))
                ybr = int(float(box_data.get('ybr')))
                ytl = int(float(box_data.get('ytl')))
                label = str(box_data.get('label'))

                #print(label)
                class_name = ['Fucus', 'Furcellaria', 'Zostera', 'None']
                label_num = class_name.index(label)

                #print(label_num)
                #https://pypi.org/project/mean-average-precision/
                gt = np.array([[xtl, ytl, xbr, ybr, label_num, 0, 0]], dtype=np.int64) #idk about the 0,0 with
                
                
                # create metric_fn
                metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=4)

                # add some samples to evaluation
                for i in range(10):
                    metric_fn.add(pred, gt)
                    
                # compute PASCAL VOC metric
                print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")

                # compute metric COCO metric
                print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.1, 0.1), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")
                
                print('\n-----------------------------\n')
                
                overall_pascal.append(metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP'])
                overall_coco.append(metric_fn.value(iou_thresholds=np.arange(0.5, 1.1, 0.1), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP'])

    overall_pascal = np.asarray(overall_pascal)
    overall_coco = np.asarray(overall_coco)
    title_pasc = method + '_lowes_' + str(lowe_ratio) + '_' + 'thresh_' + str(good_match_thresh) + '_' + 'pred_pascal_voc_valid_' + str(round(np.mean(overall_pascal), 5)) + '.csv'
    title_coco = method + '_lowes_' + str(lowe_ratio) + '_' + 'thresh_' + str(good_match_thresh) + '_' + 'pred_coco_valid_' + str(round(np.mean(overall_coco), 5)) + '.csv'
    
    print('pred_pascal_voc: %s\n' % str(round(np.mean(overall_pascal), 5)))
    print('pred_coco: %s\n' % str(round(np.mean(overall_coco), 5)))
    
    np.savetxt(title_pasc, overall_pascal, delimiter=',')
    np.savetxt(title_coco, overall_coco, delimiter=',')
               
if __name__ == '__main__':
    fd_detector()