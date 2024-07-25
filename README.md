# Shelters detection of a refugee camp on drone image with YOLOv8
This project automates detection and count of shelters in a refugee camp to estimate the population. A drone's high-def image is analyzed with YOLOv8 for detecting shelters and other categories of camp structure. The model is trained on drone images in Chad, Mozambic and the Democratic Republic of Congo (DRC). Far from perfect, the model and the script provide a good result in shelter detection of similar DRC camps. Still a lot of work need to be done to integrate more example from various country in the training dataset. Future work will explore segmentation with models like SAM.

![image](https://github.com/user-attachments/assets/894d9ca1-c232-40f2-8915-878fc03e2d04)

## Objective

The aim is to be able to automate the counting process for shelters in a refugee camp in order to estimate the population. In this case, we are in DRC, on the outskirts of Goma (North Kivu) in Elohim camp. To carry out this shelter count, a drone was used (in December 2023) to acquire a high-definition image.

In the script we will then use the [Ultralytics](https://docs.ultralytics.com/) framework , which implements Yolov8 in Python. YOLO's primary objective is to detect objects on video. In our case we're going to do it on a raster image, which is just as powerful and very fast. The aim is to detect shelters and locate them with bouding boxes, not to segment them (we need a count, not a a perfect delimitation). The next improvement will be to use bouding boxes to segment objects using a model such as [Segment Anything Model](https://docs.ultralytics.com/models/sam/) but we will explore that later.

Note that because of the size of the image, the [Slicing Aided Hyper Inference (SAHI)](https://github.com/obss/sahi) library  will be used to cut the image into pieces and submit it to Yolo.

There is a pre work of this script who as note yet be published :
The model has been trained based on the Yolo8 Nano model with Google Colab. The training data was taken from drone images of camps in Mozambique, Chad and the Democratic Republic of Congo. [Label Studio](https://labelstud.io/) was used to annotate the data and create the training dataset. At first, the annotations were done entirely by hand and then via active learning where the model is continually improved using its own output to refine its future predictions.

View of a camp around Goma:

![image](https://github.com/user-attachments/assets/8a7bee91-38c6-4351-88f7-31513a3630f5)

## Model
The model used is a fine tuning of the [Yolo8 Nano model](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes) trained via Google Colab.

Tests were carried out with the Medium and Small models, which considerably increased training time without providing much higher quality detection. This was judged by visual comparison of the results. The Nano model was chosen for its speed of training compared with the other models.

There are two model exports available: Pytorch (best.pt) and Open Neural Network Exchange (.onnx).

## Usage in QGIS
ONNX can be used directly in QGIS via the [Deepness: Deep Neural Remote Sensing](https://plugins.qgis.org/plugins/deepness/) plugin and the following parameters :
- Model type should be "Detector". We detecte bouding boxes (we do not have a segmentation model yet)
- Model file path : path to the .onnx model
- Tiles overlap should be bigger than the bigest object you want to detecte on the image
- Detector type : YOLO_Ultralytics
- Confidence : the treshold for the minimum confidence (between 0 and 1)
- IoU : [Intersection Over Union](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/) to remove overlapping detected objects (between 0 and 1)

Note that your raster unit should be in meter (e.g. : EPSG:3857 - WGS 84 / Pseudo-Mercator). 
![image](https://github.com/user-attachments/assets/2fe89d36-ba00-4a4c-b9ec-938da7e3c31a)

## Training dataset
The model was trained on a dataset including camps in **Mozambique and Chad, but above all on neighbouring camps in DRC around Goma. It is these similar camps that explain the good prediction results in this example. It is by no means certain that this model will perform as well out of the box in other camps.** From one camp to another, the category definitions presented here may no longer be relevant, affecting the quality of detection.

It should be noted that the results are good mainly for so-called "informal" shelters (makeshift shelters made of wood and plastic sheeting). This is because the training images contain a huge number of examples of these informal shelters. This dataset is then expanded (via [Albumentations](https://albumentations.ai/)). 
The other categories suffer from much poorer detection due, for the most part, to a lack of training data. For example, vehicles are regularly identified as shelters simply because the level of confidence is far too low in this category due to the lack of data. 

Here below the number of instance, after augmentation, by classes, in the training dataset :

![image](https://github.com/user-attachments/assets/6ad3dc13-c5c2-4676-8bcf-ad6884b7f7c3)

The training data is taken entirely from drone images with a GSD between 2cm/pixel to 10cm/pixel. Although the process of increasing the data includes a pass that degrades the image quality, this particularity in terms of definition suggests that the model will not perform well on images with a lower resolution (>10cm/pixel).

## Categories
When the training datasets were first created, only informal shelters were annotated. 
. After some thought, several categories were added. Unfortunately, the lack of examples in some of these categories has resulted in a mediocre detection confidence. The definition of the category and the training dataset need to be improved.

Here are the categories and their definitions:
* Building: A structure with a complex shape and a solid roof. Surface mainly above 50m²

  ![image](https://github.com/user-attachments/assets/6c9f9d20-9308-42cc-bbe7-8504a2b38747)


* Dwelling: A simple, regularly shaped structure with a basic corrugated iron roof, often rusted. Surface mainly between 10 to 50 m²

  ![image](https://github.com/user-attachments/assets/4559ec5a-4cba-414e-a085-873d08f87987)
  ![image](https://github.com/user-attachments/assets/d11f7981-f674-40e2-ae6e-1b756024ac67)

    
* Shelter Informal: A small size shelter (few meter square), irregularly shaped construction made with plastic sheeting and a plastic roof. Surface mainly between 2 to 10 m²

  ![image](https://github.com/user-attachments/assets/9e949b63-58ea-4d5d-b7b9-68e5df38f5d6)
  ![image](https://github.com/user-attachments/assets/8dceb4c1-1828-4eb5-80d8-81aefe462b41)

    
* Shelter Formal: A small, rectangular or square construction made with plastic sheeting, often with a corrugated iron roof. This one is tricky because it can be easily confuse with informal shelter used in marketplace. Surface mainly between 15 to 25 m²

  ![image](https://github.com/user-attachments/assets/b6e2e983-e2e8-4836-8b43-4f0c68d1a6a8)

  
* Under Construction : A structure with visible wood sticks, maybe be incompletely covered by plastic sheeting or just the foundations of a structure. Note that this category may mix under construction shelter and destroyed or vanalised shelters. There is one exception to the definition : In some structured camps, when the formal shelters are under construction, 6 or 8 holes are digged into the ground. They are classified as "Under construction"

  ![image](https://github.com/user-attachments/assets/3eb53bf0-cb5d-45ab-8176-df2aadac0c7e)
  ![image](https://github.com/user-attachments/assets/a5282b62-23b1-4519-b290-3b2b5d8fceb4)
  ![image](https://github.com/user-attachments/assets/fab49ab7-e07e-4130-bbae-2ad0771ba62a)
  ![image](https://github.com/user-attachments/assets/61c3ceb8-39d6-4363-a95f-3a532ab7569d)
  ![image](https://github.com/user-attachments/assets/2227cf46-adf4-4485-8359-89a680acbca8)
  ![image](https://github.com/user-attachments/assets/464b5fd0-2265-4b93-a699-649c6eac327c)
  ![image](https://github.com/user-attachments/assets/038c9231-563b-447f-a491-fbc067f451d8)

  
* Construction hole : One hole in the ground without constructed structure around. Often a sign of a new latrine

  ![image](https://github.com/user-attachments/assets/89aa1417-c6f1-4929-b10c-17c7433cef7e)
  ![image](https://github.com/user-attachments/assets/3be24a93-7a4e-4686-ab90-dceb35ad9b07)
  ![image](https://github.com/user-attachments/assets/75d96b93-9955-40c5-9629-39c7c41f1d2c)

* Car: A vehicle with four or more wheels. Can be a car or a truck. Note that this category can easily be improved with example from the web.

  ![image](https://github.com/user-attachments/assets/62723b92-f70f-42a9-87cc-f8b124c6e0be)
  ![image](https://github.com/user-attachments/assets/2eba4282-0d45-44ff-80c4-391aabe1812b)
  ![image](https://github.com/user-attachments/assets/f7590f38-c3e6-4ac1-a250-58f09f0c28c3)

  
* Hut: A small construction, either rectangular or round, with a straw roof.

  ![image](https://github.com/user-attachments/assets/06292012-0dbb-4bcd-8739-df25d5977dc4)
  ![image](https://github.com/user-attachments/assets/47c3bb8b-a195-4715-ad0d-f657edf8670b)
  ![image](https://github.com/user-attachments/assets/09946bde-e003-4fde-b65f-8761632e6f8c)

  
* Latrine: A dwelling featuring elements typical of a latrine, such as a chimney, double door, or concrete slab. This category need to be remove because it is too complicated for the model to detecte it. Often it is mixed with a simple dwelling. The elements to clearly identify it from the sky are often difficult to see. it's often or the position of the latrine (a bit away from shelters) or the site visit that identifies the dwelling as a latrine, rather than just the look of the latrine from the sky.
  
  ![image](https://github.com/user-attachments/assets/9ff381fe-bc68-4df9-a6c7-0755f3265285)
  ![image](https://github.com/user-attachments/assets/e9fde3aa-9ac2-47a4-a3f6-0dac61315b56)

  
* Latrine Plastic Sheeting : squares or rectangle latrine without a roof, made of wood and plastic sheeting. Rarely more than 6 or 8 m²

  ![image](https://github.com/user-attachments/assets/b73db629-5979-4bcd-9b93-89ea323b3f91)
  ![image](https://github.com/user-attachments/assets/d3e1be80-e15c-4a93-bc6d-dc24de6fb1ca)
  ![image](https://github.com/user-attachments/assets/81c92e83-9842-4b87-b14c-67b694b4b163)

  
* Water Source: Mutliple taps where people gather to collect water. Often with water hoses and/or hand pumps. Only few examples of this in the datasets. Most of the time there is a lot of people with buckets around.

  ![image](https://github.com/user-attachments/assets/7dc6b866-f703-4cf2-b54f-80f7ea36e2c6)
  ![image](https://github.com/user-attachments/assets/c31762df-91ea-401a-8e0c-2ce4492a1b17)
  ![image](https://github.com/user-attachments/assets/bacaa7e0-e667-4a36-948e-43ea52a3eb52)

  
* Tank: Any (water) reservoir.

  ![image](https://github.com/user-attachments/assets/fbc62440-0f27-41ab-9c8a-777ed9a265e0)
  ![image](https://github.com/user-attachments/assets/e9569c61-9e91-4130-bbae-a4149c939ac4)


* Tent : A simple tent, often black and white, with visible underlying metal support uprights. Commonly used in camps as a temporary health facility or storage.

  ![image](https://github.com/user-attachments/assets/002808a9-d071-429a-8af0-147d14c89e32)
  ![image](https://github.com/user-attachments/assets/06b76ec7-4cf3-426e-b3ef-626815fb98e4)


* Umbrella : Simple umbrella, often use along the road as a small shop

  ![image](https://github.com/user-attachments/assets/2ec2a279-15e8-4143-9726-5fcaf860d5c5)

## Hardware
Yolo runs easily on a laptop with standart GPU. For this example, the scipt for infering was run on a NVIDIA T1200 Laptop GPU, 4096MiB without any issue.

## Improvement and next steps
As mentioned, there is still considerable work ahead, primarily including:
- Enhancing the model by incorporating more examples from various countries and categories.
- Exploring segmentation techniques with models such as SAM.
- Implementing the script within a QGIS toolbox.
- Addressing additional minor improvements.
 
## Contribution
I’ll be glad to collaborate with others on this project. If you have ideas, improvements, or would like to contribute in any way, please feel free to reach out. I welcome all forms of collaboration.

## Thanks
Thanks to Andrea Cippà, siteplanner and drone pilote, for having provided a lot of camps images! Here is a link to his [Youtube Channel](https://www.youtube.com/@AndreaCippa-Siteplanning)

--------------------------------------------------------------------------------------------

Thank you for your interest and contributions!
