# Smart Traffic Management and Vehicle Detection

A beginner-friendly traffic video processing project using Python and OpenCV. It detects moving vehicles from a road video, draws bounding boxes, and shows the vehicle count for each frame.

## Project structure

```text
Smart-Traffic-Management-and-Vehicle-Detection/
├── src/
│   └── traffic_detection.py
├── input/
├── output/
├── requirements.txt
└── README.md
```

## Run

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
python src/traffic_detection.py --input input/traffic.mp4
```

The processed video is saved as `output/detected_traffic.mp4`.

## Note
This repository is a clean starter implementation derived from the project description on the resume. Add your own training/model files and research notes when available.
