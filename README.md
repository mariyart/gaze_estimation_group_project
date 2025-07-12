# [ECCV 2024] 3DGazeNet: Generalizing Gaze Estimation with Weak-Supervision from Synthetic Views

3DGazeNet is a general gaze estimation model which can be **directly employed in novel environments without adaptation**.
In 3DGazeNet we leverage the observation that head, body, and hand pose estimation benefit from revising them as dense 3D coordinate prediction, and similarly express gaze estimation as regression of dense 3D eye meshes. In addition, we employ a diverse set of unlabelled, in-the-wild face images to boost gaze generalization in real images and videos, by enforcing multi-view consistency constraints during training.


<p align="center">
  <img src="assets/teaser_1_2.png" height="200" title="teaser1">
  <img src="assets/teaser_2_2.png" height="200" title="teaser2">
</p>

https://github.com/Vagver/dense3Deyes/assets/25174551/4de4fb76-9577-4209-ba07-779356230131


## Demo

For a demo of 3DGazeNet on videos and single images visit the [demo folder](demo).


## Installation

To create a conda environment with the required dependences run the command: 

```
$ conda env create --file env_requirements.yaml
$ conda activate 3DGazeNet
```

## Download models

Download the data directory contatining pre-trained gaze estimation models from [here](https://drive.google.com/file/d/1mYvKRJGS8LY5IU3I8Qfvm-xINQyby1z5/view?usp=sharing). Extract and place the data folder in the root directory of this repo.

## Inference

For the final project of the group project, the 3DGazeNet was modified to also detect drowsiness, blinking, yawning and depth. The final script is demo\inference_video_integrated.py and is run by:

```
inference_video_integrated.py
```

in the demo folder. It's necessary to change the video path in the script itself.


## Citation
If you find our work useful in your research, please consider to cite our paper:
```
@inproceedings{ververas20253dgazenet,
  title={3DGazeNet: Generalizing 3D Gaze Estimation with Weak-Supervision from Synthetic Views},
  author={Ververas, Evangelos and Gkagkos, Polydefkis and Deng, Jiankang and Doukas, Michail Christos and Guo, Jia and Zafeiriou, Stefanos},
  booktitle={European Conference on Computer Vision},
  pages={387--404},
  year={2025},
  organization={Springer}
}
```
