
# Human Detection

## Instalation

`python version 3.10`

`pip install -r requirements.txt`


## Run

### Usage

```
python main.py [-h] [-r] [-t THRESHOLD] [-f FILE_PATH] [-fps FPS_CAP] [-wc WEB_CAM] [-hw]

CLI for human detection

options:
  -h, --help            show this help message and exit
  -r, --use_rknn        Enable RKNN usage
  -t THRESHOLD, --threshold THRESHOLD
                        Detection threshold value
  -f FILE_PATH, --file_path FILE_PATH
                        Path to video file, setting this will ignore web_cam argument
  -fps FPS_CAP, --fps_cap FPS_CAP
                        FPS cap (optional)
  -wc WEB_CAM, --web_cam WEB_CAM
                        Webcam number, 0 for first webcome (optional)
  -hw, --hide_window    Show the video/cam (affects performance)
```

### Examples

*File:* `python main.py -f "./resource/4.m4v" `

*Web cam:* `python main.py -wc 0`

*FPS capped at 10:* `python main.py -f "./resource/4.m4v" -fps 10`

*Without showing video:* `python main.py -f "./resource/4.m4v" -hw`

*FPS capped at 10 hiding the video:* `python main.py -f "./resource/4.m4v" -fps 10 -hw`

