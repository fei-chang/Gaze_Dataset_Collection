# Gaze Dataset Collection

This repo contains general information on the datasets we collected for gaze-following and gaze pattern recognition tasks. It also introduce tools and some helper functions that we use in the collection and annotation process.

## Collected Dataset
<details>
<summary> GP-Static(Published)</summary>

### GP-Static(Published)
| # Vids | fps | multi-view | length per vid | # annotations | headbox | gaze point | gaze patterns | 
|:------:|:---:|:----------:|:--------------:|:-------------:|:-------:|:----------:|:-------------:|
|   370 |  25  |     ❌    |   3-15 seconds  | 169,364       |  ✅    |   ❌       |     ✅      |


**Source:** The dataset contains videos sampled from two existing dataset: [UCO-LAEO](https://github.com/AVAuco/ucolaeodb/) and [GazeCommunication](https://github.com/LifengFan/Human-Gaze-Communication)

**Location:** The dataset currently save under `data/GazeFollow_ours/GP_static` in the public Nas of VIPL.

**Additional Remarks:** The dataset is designed for the static gaze pattern classification task. Published with [Gaze Pattern Recognition in Dyadic Communication](https://dl.acm.org/doi/abs/10.1145/3588015.3588411).

</details>




<details>
<summary> DyadicGaze (Under Construction)</summary>

###  DyadicGaze (Under Construction) 
| # Vids | fps | multi-view | length per vid | # annotations | headbox | gaze point | gaze patterns | eyes|
|:------:|:---:|:----------:|:--------------:|:-------------:|:-------:|:----------:|:-------------:|:-------------:|
|   325  | 25 |    ❌    | 6-20 seconds |    191,384   |  ✅   |   ✅      |    ✅    | ✅  |



**Source:**  The dataset contains videos sampled from youtube, pexels and envatos.

**Location:** The dataset currently save under `data/GazeFollow_ours/DyadicGaze` in the public Nas of VIPL. Annotations can be downloaded from [google drive](https://drive.google.com/file/d/1exCjzPXUDy65qv6HpCVcJL1eDssam_T8/view?usp=sharing)

**Additional Remarks:**

The dataset is designed for the static and dynamic gaze pattern classification task. Detailed Information
|          | Train | Test  | 
|----------|-------|-------|
| #vids    | 263   | 62    |
| #frames  | 76099  | 19593 |
|#Share|28370 (18.64%)|8332 (21.26%)|
|#Mutual|23818 (15.65%)|7016 (17.90%)|
|#Single|22419 (14.73%)|5886 (15.02%)|
|#Miss|22423 (14.73%)|5884 (15.02%)|
|#Void|55168 (36.25%)|12068 (30.80%)|
</details>

<details>
<summary> ShanghaiASD_20230301 (Under Construction)</summary>
  
### ShanghaiASD_20230301 (Under Construction)

| # Vids | fps | multi-view | length per vid | # annotations | headbox | gaze point | gaze patterns | 
|:------:|:---:|:----------:|:--------------:|:-------------:|:-------:|:----------:|:-------------:|
|  17(out of 40)  |  30  |     ✅    |  6-20 minutes   |    633,900    |  ✅    |   ✅       |      ❌    |

**Location:** The dataset currently save at `10.29.0.195:/affect/D/Data/data/ShanghaiASD/20230301`

**Additional Remarks:** Currently there are multi-view annotations on 7 instances, under each instance, 2-4 views from different cameras are available.

</details>

<details>
<summary> ShanghaiASD_20230531 (Under Construction)</summary>
### ShanghaiASD_20230531 (Under Construction)
</details>


## Collection Tools

<details>
<summary> Head Detection </summary>

In repo [YOLOv8_head_detector](https://github.com/Abcfsa/YOLOv8_head_detector), is a tool for detecting heads in frames.
In repo [Tool_head_detector](https://github.com/fei-chang/Tool_head_detector/tree/main), is an old version of head detector based on yolov3 used for detecting heads in frames.

</details>


<details>
<summary> Person Tracking </summary>
  
In file: `PersonTracker.py`, used for tracking the target person from unlabeled heads across frames.

Usage:
```python
person_tracker = PersonTracker()
raw_head_detections = 'path_to_raw_annotations/raw_detections.txt'
person_tracker.load_from_files(raw_head_detections, frame_dir)
target_person = 'kid'
person_tracker.track_person(target_person)
df = person_tracker.get_person_df(target_person)

person_tracker.release()
```

Track multiple people
```python
person_tracker = PersonTracker()
raw_head_detections = 'path_to_raw_annotations/raw_detections.txt'
person_tracker.load_from_files(raw_head_detections, frame_dir)
target_IDs = ['p1', 'p2', 'p3']
for personID in target_IDs:
  person_tracker.track_person(personID)
df = person_tracker.get_full_df()

person_tracker.release()
```

 **Note on input file format: raw_detections.txt** 
 - :x: No column header, entries are organized as `['frameID', 'label', 'xmin', 'ymin', 'xmax', 'ymax']`
 - :x: No index column
 - Entries of 'xmin', 'ymin', 'xmax', 'ymax' are all in 0-1 scale
</details>

<details>
<summary> Annotation Tool </summary>
  
At [CVAT Online](http://43.138.12.230:8080/), used for create annotations on videos. Some helper functions can be found at `cvat_utils.py`

Usage:
1. Conver exported cvat annotation file to dictionary. (Exported format: CVAT for video 1.1)
```python
from cvat_utils import cvat2dict

info_dict = cvat2dict('path/to/cvat_annotations.xml')

```
**Note**
- 在使用网页（尤其是上传视频数据）时，建议关闭VPN，会卡顿。
- Advanced Configuration中，建议不要选取 Use zip/video chunks 选项，该选项容易造成在视频在个别帧数间卡顿现象。
- Advanced Configuration中，Segment Size控制每个Job的帧数，Chunk size则控制系统在分包时，每一个分开的压缩包中图像的数量。对于过长的视频，建议选取Segment Size进行控制每个Job的时长，同时每个Segment的时长应设为Chunck size的整数倍。
- 在进行标注时，遇到卡顿现象可以通过'F'键前进一帧，强制加载下一帧跳过卡顿。
- 在进行标注时，不建议大范围拖动进度条，会卡顿。
- 更多使用指南以及操作可见CVAT[官方指南](https://opencv.github.io/cvat/docs/getting_started/) 与[Repo](https://github.com/opencv/cvat).
  
</details>
