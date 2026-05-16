# Weights

Model weights are not tracked by Git. Download the trained weights from Google Drive:

| Model | File | Description | Download |
|------|------|-------------|----------|
| WLP-YOLO | `wlp_yolo.pt` | Full, unpruned WLP-YOLO model reported in the paper | [Google Drive](https://drive.google.com/file/d/1OBbVycS4jdrDyo5VEel0fU-ODW4s9tnk/view?usp=sharing) |
| WLP-YOLO (pruned) | `wlp_yolo_pruned.pt` | Structured-pruned WLP-YOLO model | [Google Drive](https://drive.google.com/file/d/1eidb0EtrBtzzsaKMmaf-IaNRDwLP3Ppl/view?usp=sharing) |

Place them as:

```text
weights/wlp_yolo.pt
weights/wlp_yolo_pruned.pt
```

The original scripts currently use paths such as `best.pt` and `GSA.pt`; either place the downloaded weights at those paths or update the script defaults.
