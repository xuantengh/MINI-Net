# [MINI-Net: Multiple Instance Ranking Network for Video Highlight Detection](https://arxiv.org/abs/2007.09833)

This repo contains source code for our ECCV 2020 work **MINI-Net: Multiple Instance Ranking Network for Video Highlight Detection**. Our model is implemented under Pytorch.

![image-20200926112429908](asset/demo.png)

## Prerequisites

1. Pytorch 1.4 +
2. numpy
3. tqdm
4. moviepy
5. cv2

## Preparing Data

- step1: split audio and video from origin data:

  First, group your custom dataset into different directories by their category like this:

  ```
  .
  |-- base_jump
  |   |-- 1.mp4
  |   |-- 2.mp4
  |   |-- 3.mp4
  |   |-- ...
  |-- bike_polo
  |   |-- 1.mp4
  |   |-- ...
  |-- eiffel_tower
  |   |-- 1.mp4
  |   |-- ...
  |-- excavators_river_cross
  |   |-- ...
  ```

  Then run `youtube_splitAudio` method in `visual-audio fusion/tools.py` to split audio file from origin video. Parameter `path` is the path to videos, while the `savepath` is the path to save the splited `.wav` audio file.

- step2: extract audio feature with pretrained model and store in file:

  ```bash
  cd audio_process
  python AudioCNN.py --audio_path path/to/wav_files --save_path path/to/save/audio/features
  ```

- step3: extract video feature with pretrained model and store in file:

  ```bash
  cd video-classifcation-3d-cnn-pytorch
  CUDA_VISIBLE_DEVICES=1 python main.py \
  --video_root path/to/videos/belongs/to/same/category \
  --savename path/to/save/video/features \
  --model ./resnet-34-kinetics.pth --mode feature
  ```

  You must install `ffmpeg` to use C3D to extract visual features. See more details at their readme:`video-classifcation-3d-cnn-pytorch/README.md`.

- step4: get video length:

  ```bash
  cd audio_process
  python getDuration.py --audio_path path/to/wav_files --duration_path path/to/save/video/lengths
  ```

## Train

Change directory to `src` and run following code:

```bash
CUDA_VISIBLE_DEVICES=1 python trainMIL.py --dataset youtube --domain gymnastics \
	--train_path /home/share/Highlight/proDataset/TrainingSet/ \
	--test_path /home/share/Highlight/proDataset/DomainSpecific \
	--topk_mAP 1 --FNet MILModel3 --HNet HNet1 --AM AttentionModule_1 \
	--DS MILDataset --AHLoss AdaptiveHingerLoss \
	--short_lower 10 --short_upper 40 --long_lower 60 --long_upper 60000 --bagsize 60 
```

Parameters:

- `CUDA_VISIBLE_DEVICES`: specify GPU Id for training
- `dataset`: choose dataset, alternatives: youtube, tvsum, cosum
- `domain`: choose target domain in given dataset, *e.g.*, gymnastics for youtube dataset
- `train_path`: extracted feature file for training, mentioned above
- `test_path`: extracted feature file for testing
- `topk_mAP`: specify test metric, 1 or 5 in our paper
- `FNet`: which model to use to predict highlight score for each segment in video, in our paper: `MILModel10`
- `AM`: which model to fuse visual feature and audio feature, in our paper: `AttentionModule_1 `
- `DS`: dataset model: in out paper: `MILDataset`
- `AHLoss`: hinger loss used in our paper

See `visual-audio fusion/opts.py` for details of data selection hyper-parameters.

Note: In our code, the directory  `train_path`  in the parameter must include:

1. visual feature files generated in step 3
2. audio feature files generated in step 2
3. video length files generated in step 4

For example, there contains 3 files for category PK belongs to TVsum dataset in that directory:

```bash
PK_1s.npy											// visual features for video with categoty PK
PK_audio_edited_nopost.npy		// audio features
PK_duration.npy								// duration information for each valid video
```



## Main Results

- Youtube:

| Topic      | mAP    |
| ---------- | ------ |
| dog        | 0.5816 |
| gymnastics | 0.6165 |
| parkour    | 0.7020 |
| skating    | 0.7217 |
| skiing     | 0.5866 |
| surfing    | 0.6514 |

- TVsum:

| Topic | top-5 mAP |
| ----- | --------- |
| VT    | 0.8062    |
| VU    | 0.6832    |
| GA    | 0.7821    |
| MS    | 0.8183    |
| PK    | 0.7807    |
| PR    | 0.6584    |
| FM    | 0.5780    |
| BK    | 0.7502    |
| BT    | 0.8019    |
| DS    | 0.6551    |

- CoSum:

| Topic | top-5 mAP |
| ----- | --------- |
| BJ    | 0.8450    |
| BP    | 0.9887    |
| ET    | 0.9156    |
| ERC   | 1         |
| KP    | 0.9611    |
| MLB   | 0.9353    |
| NFL   | 1         |
| NDC   | 0.9536    |
| SL    | 0.8896    |
| SF    | 0.7897    |

## Reference

If you find our work helpful in your research, please cite our paper via:

```
Bib:
@inproceedings{hong2020mini,
title={MINI-Net: Multiple Instance Ranking Network for Video Highlight Detec- tion},
author={Hong, Fa-Ting and Huang, Xuanteng and Li, Wei-Hong and Zheng, Wei-Shi},
booktitle={European Conference on Computer Vision},
year={2020}
}
```

