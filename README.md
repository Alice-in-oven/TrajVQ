# TrajVQ

## Requirements

```
pip install -r requirements.txt
```

## Data

GeoLife and Porto come from ConvTraj (https://github.com/Proudc/ConvTraj) with the standard ConvTraj
splits. T-Drive comes from the processed data of SIMformer
(https://github.com/SUSTC-ChuangYANG/SIMformer); convert it into the same layout with

```
python prepare_tdrive_split.py --input_root T-Drive --output_root data/tdrive
```


## Training

Configs are in `configs/<dataset>/<measure>_{continuous,vq}.json` for `geolife`, `porto`, `tdrive` and
`dtw`, `haus`, `dfd`. The example below is GeoLife Hausdorff.

Step 1 trains the encoder without quantization:

```
python train.py --config configs/geolife/haus_continuous.json --data_root DATA \
    --output_dir runs/geolife_haus_continuous --device cuda:0
```

Step 2 fits the OPQ teacher on the continuous training embeddings:

```
python fit_opq_teacher.py --config configs/geolife/haus_vq.json \
    --embeddings runs/geolife_haus_continuous/embeddings/train.npy \
    --output_dir runs/geolife_haus_teacher
```

Step 3 trains TrajVQ from the step 1 encoder and the step 2 teacher:

```
python train.py --config configs/geolife/haus_vq.json --data_root DATA \
    --output_dir runs/geolife_haus_vq --device cuda:0 \
    --backbone runs/geolife_haus_continuous/model.pt \
    --teacher_dir runs/geolife_haus_teacher
```

GeoLife DTW and T-Drive DTW skip steps 1 and 2 and train the VQ config directly:

```
python train.py --config configs/geolife/dtw_vq.json --data_root DATA \
    --output_dir runs/geolife_dtw_vq --device cuda:0
```

Each run writes `model.pt`, `train_log.json` and `metrics.json` to `--output_dir`.

## Evaluation

Training ends with an evaluation of the final model. To evaluate a saved model, pass it with
`--checkpoint`:

```
python train.py --config configs/geolife/haus_vq.json --data_root DATA \
    --output_dir runs/geolife_haus_eval --device cuda:0 --checkpoint runs/geolife_haus_vq/model.pt
```
