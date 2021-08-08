# Eye of ML

### Requirements WGET
Windows, download WGET from https://eternallybored.org/misc/wget/ (wget.exe) and move it to PATH environment (or make that path a PATH). Linux, there are lots of guide out there which should be easy to search. 


Simple program to run yolov5 (large) object detection with video camera. 

Note this requires a stronger graphics card to run (tested on GTX1080Ti). 

It is recommended to **not** run `pip install -r requirements.txt` directly but to make them yourselves. Particularly, one is not sure how to get distributable from pytorch into requirements.txt hence requires you to install pytorch with cuda yourself, using the code defined from [Pytorch repository](https://pytorch.org/get-started/locally/). Particularly, choose the one with CUDA support according to your OS and CUDA version.  

Run the program with: `python -m render` or `python3 -m render` or perhaps `python -m render.py` or `python3 -m render.py` depending on which one works for your os and which one doesn't raise an error. 

Credits to https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/tree/master for the EfficientDet implementations. 
Some of the folders and utilities (python files) in this directory is a direct fork from the files
so the model could be loaded and use. 

Thanks to https://ultralytics.com/ for their color palette usage in this project.

To use EfficientDet, it is required to first run `efficientdet.bat` for Windows. 
Linux support may come later (or change `.bat` to `.sh` and it will run since it's the same code anyways). 

Note that performance-wise, Yolov5 uses less CPU so if you have a sufficiently strong GPU (about 40% utilization on GTX 1080 Ti, so equivalent strength) with perhaps sufficiently high compute capability (some program might not run on lower compute capability, depending on support. Search through PyTorch's pages for more information), it should be able to run very smoothly. 

For EfficientDet, GPU is a requirement. This is because the for EfficientDet D0, inference takes about 25-30 ms, and preprocessing takes about 15-20 ms, a non-bearable amount, compared to Yolov5 which altogether inclusive only takes 20-30 ms. 

Regarding EfficientDet's benchmark from [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch), it is unsure how the fps are calculated. However, one main reason fps suffers is due to one of the operations on CPU (i.e. `postprocess_video`, and moving this operation to entirely on GPU reduces the suffering significantly (from 15ms for the default value, preprocess a 640x480x3 frame; to 2ms on GPU; though PyTorch on CPU due to converting to Torch will take 16-17ms, longer than with numpy or pure python). Now, irregardless of which variant of EfficientDet you would like to try: d0, d4, or d8 chosen here, it will run at the **same fps**. The only difference is d0 uses less than 10% of GPU, d4 uses about 40%, d8 uses about 70-80%. For other issues please check the [wiki](https://github.com/Wabinab/eye_of_ml/wiki). 

### Fast Neural Style Transfer
Taken from https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/style_transfer_inference.ipynb

GPU is required to run this program. 

### Canny Edge Detection
Taken from: https://github.com/DCurro/CannyEdgePytorch/blob/master/canny.py

Can run (quite) smoothly on CPU. GPU will speed it up by smoothing the frame. With CPU only, frame rate have some degradation but still runnable (at around 15-30 fps for 8vCPUs 1.8GHz). 
