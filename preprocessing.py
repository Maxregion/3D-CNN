
# coding: utf-8

# In[10]:

#%matplotlib inline
import numpy as np
import pandas as pd
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2


INPUT_FOLDER = 'stage1\\'
STORE_FOLDER = 'stage1\\'
data_folder='stage1\\'
OUT_FOLDER='de_nosie\\preprocess_turi_slices_label'
SHAPE=[64,32,32]

patients = os.listdir(INPUT_FOLDER)
patients.sort()
patient_information=pd.read_csv("D:\\stage1_labels.csv")
patient_list=patient_information.id.values.tolist()
patient_length=len(patient_information)

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[3,3,3]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    #print(new_real_shape)
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

def resize_image(patient_pixels,new_shape):
    tmp_slices_images=np.zeros(new_shape,np.uint8)
 
    shape=patient_pixels[0].shape
    distance=len(patient_pixels)/new_shape[0]
    print(new_shape[0])
    for i in range(new_shape[0]):
        emptyImage = np.zeros(shape, np.uint8)
        emptyImage=patient_pixels[int(i*distance)]
        tmp_slices_images[i]=cv2.resize(patient_pixels[int(i*distance)],shape)
    return  tmp_slices_images

def reshaping(image,shape):
    new_image=np.zeros(shape)
    old_shape=image.shape
    new_shape_stride=[]
    for i in range(len(shape)):
        new_shape_stride.append(old_shape[i]/shape[i])
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                new_image[x][y][z]=image[int(x*new_shape_stride[0])][int(y*new_shape_stride[1])][int(new_shape_stride[2]*z)]
                
    
    return new_image

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

def getting_patient_conditions(id):
    for i in range(patient_length):
        if patient_information.id[i]==id:
            return patient_information.id[i],patient_information.cancer[i]
        
def slice_images(patient_pixels,store_path,number):
    tmp_slices_images=np.zeros((number,64,64), np.uint8)
    
    if not os.path.exists(store_path):   
        os.makedirs(store_path)
    shape=patient_pixels[0].shape
    distance=len(patient_pixels)/number
    
         
    for i in range(number):
        emptyImage = np.zeros(shape, np.uint8)
       
        emptyImage=patient_pixels[int(i*distance)]
              
        tmp_slices_images[i]=resize_images(store_path,emptyImage,i)
    
    
    return tmp_slices_image

def new_extracted_slice_images():
    whole_data=[]
    patient_id=[]
    patient_condition=[]
    patient_index=pd.DataFrame(columns=["ID","cancer"])
    count=1
    out_count=0
    percent=0
    total_files=len(patients)
    path=OUT_FOLDER
    shape=SHAPE
    for i in patients:
        
        if i in patient_list:
            patients_path=INPUT_FOLDER + i
            store_path=STORE_FOLDER + i
            try:
                patients_information=load_scan(patients_path)
                patient_pixels = get_pixels_hu(patients_information)
                patient_pixels_resampled, spacing = resample(patient_pixels, patients_information, [3,3,3])
                
                segmented_lungs = segment_lung_mask(patient_pixels_resampled, False)
                segmented_lungs_fill = segment_lung_mask(patient_pixels_resampled, True)
                patient_slices=segmented_lungs_fill - segmented_lungs
                patient_slices=normalize(patient_slices)
                
                patient_slices=zero_center(patient_slices)
                resized_patient_slices=reshaping(patient_slices,shape)
                
                whole_data.append(resized_patient_slices)
                id,cancer=getting_patient_conditions(i)
                patient_id.append(id)
                
                patient_condition.append(cancer)
            except Exception  as e:
                print(store_path,e)
       
            
        count=count+1
        if int(count/total_files*100) > percent:
            
            percent = int(count/total_files*100)-1
              
            print (percent,"%")
               
        if count %50==0:
            patient_index.ID=patient_id
            patient_index.cancer=patient_condition
            patient_index.to_csv(path+str(out_count)+".csv",index=False)
            np.save(path+str(out_count),whole_data)
            whole_data=[]
            patient_id=[]
            patient_condition=[]
            patient_index=pd.DataFrame(columns=["ID","cancer"])
            out_count=out_count+1
            print("SAVING at ",path+str(out_count))
    patient_index.ID=patient_id
    patient_index.cancer=patient_condition
    patient_index.to_csv(path+str(out_count)+".csv",index=False)
    np.save(path+str(out_count),whole_data) 
#extract lung  model  from dicom data
new_extracted_slice_images()

path="de_nosie\\"
merge_path="merge\\preprocess_turi_slices_image"
merge_path_label="merge\\preprocess_turi_slices_label.csv"
shape=[64,32,32]

#merge data
class merge:
    def __init__(self,path,merge_path,merge_path_label,shape):
        self.path=path
        self.merge_path=merge_path
        self.merge_path_label=merge_path_label
        self.shape=shape
    def merge(self):
        files=[]
        file_len=0 
        included_extenstions=['npy']
        file_names = [fn for fn in os.listdir(self.path)if any(fn.endswith(ext) for ext in included_extenstions)]
        for file in file_names:
            tmp_image_batch_path=path+file
            tmp_batch=np.load(tmp_image_batch_path)                        
            file_len+=len(tmp_batch)
            files.append(tmp_batch)
            #print(tmp_image_batch_path)
        shape=[file_len,self.shape[0],self.shape[1],self.shape[2]]

        new_image_batch=np.zeros(shape)

        start=0

        for i in range(len(files)):
            for j in range(len(files[i])):
                new_image_batch[start]=files[i][j]
                start+=1
                #print(start)


        np.save(self.merge_path,new_image_batch)
        return new_image_batch
    
    
    def merge_label(self):
        files=[]
        #batch_image_filename=os.listdir(outpath)
        file_len=0 
        Datafram=pd.DataFrame(columns=["ID","cancer"])
        included_extenstions=['csv']
        file_names = [fn for fn in os.listdir(self.path)if any(fn.endswith(ext) for ext in included_extenstions)]


        for file in file_names:
            tmp_label_path=path+file
            tmp_batch=pd.read_csv(tmp_label_path)   
            #print(tmp_label_path)
            file_len+=len(tmp_batch)
            files.append(tmp_batch)
        result=pd.concat(files)
        result.to_csv(self.merge_path_label,index=False)
        
    def merge_all(self):
        self.merge()
        self.merge_label()
        
merge=merge(path,merge_path,merge_path_label,shape)
merge.merge_all()

