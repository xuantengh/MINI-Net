revise `train.py` and other modules relevant to it.

just run `train.sh` to train for the domain *skating* in dataset *TVsum*. 


### skating.npy: defaultDict(list)
-   key: 文件名， e.g., 201493439749854.mp4
-   value: List
    -   element: dict
        -   segment:[1,15]
        -   features: [12,232,343,42,12,3,343,43,42,3,,43,3] #512 dim
### skating_duration.npy: defaultDict(list)
-   key: 文件名， e.g., 201493439749854.mp4
-   value: float  #156
### skating_audio.npy: defaultDict(list)
-   key: 文件名， e.g., 201493439749854.mp4
-   value: List
    -   element: dict
        -   segment:[1,15]
        -   features: [12,232,343,42,12,3,343,43,42,3,,43,3] #20 * 23