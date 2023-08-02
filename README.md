# Gaze Dataset Collection

## Collected Dataset
<details>
<summary> GP-Static(Published)</summary>

### GP-Static(Published)
| # Vids | fps | multi-view | length per vid | # annotations | headbox | gaze point | gaze patterns | 
|:------:|:---:|:----------:|:--------------:|:-------------:|:-------:|:----------:|:-------------:|
|   370 |  25  |     ❌    |   3-15 seconds  | 169,364       |  ✅    |   ❌       |     ✅      |


**Source:** The dataset contains videos sampled from two existing dataset: [UCO-LAEO](https://github.com/AVAuco/ucolaeodb/) and [GazeCommunication](https://github.com/LifengFan/Human-Gaze-Communication)

**Location:** The dataset currently save at `/home/changfei/X_Nas/data/GazeFollow_ours/GP_static`

**Additional Remarks:** The dataset is designed for the static gaze pattern classification task. Published with [Gaze Pattern Recognition in Dyadic Communication](https://dl.acm.org/doi/abs/10.1145/3588015.3588411).

</details>




<details>
<summary> GPCS (Under Construction)</summary>

###  GPCS (Under Construction) 
| # Vids | fps | multi-view | length per vid | # annotations | headbox | gaze point | gaze patterns | 
|:------:|:---:|:----------:|:--------------:|:-------------:|:-------:|:----------:|:-------------:|
|     |  |    |  |       |   |       |      |



**Source:**  The dataset contains videos sampled from youtube, pexels and envatos.

**Location:** The dataset currently save at 

**Additional Remarks:**

The dataset is designed for the static and dynamic gaze pattern classification task.

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
  
In repo [Tool_head_detector](https://github.com/fei-chang/Tool_head_detector/tree/main), used for detect heads in frames.

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
