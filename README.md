# Image Colorization



## Links to Download trained models:

[Results](https://drive.google.com/drive/folders/1mPM673EesECNAtnXNATPIfve0hhQnuSz?usp=sharing) : 



## Dependencies

- scikit-image
- pytorch
- matplotlib 
- numpy 


## Folder Structure

├── models/                                   
│   └── colorization_deploy_v1.py                          # our pytorch model class
├── notebooks/                        
│   ├── sun.jpg                                            # test image
│   └── test.ipynb                                         # test script (lab to rgb conversion, model import, etc)
├── .gitignore
├── README.md
├── train.py                 
└── utils/
    ├── color_quantization.py                              # defines NNEncode, to encode and decode color in discrete space of dim 313, defined in the article
    ├── data.py                                            # dataloader  script
    ├── model_to_video_script.py                           # script to write videos using a trained model
    └── weights.py                                         # class that compute weights in balanced class loss

