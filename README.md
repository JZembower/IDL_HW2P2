# HW2P2: Face Classification & Verification with CNNs

## Executive summary
We train a CNN from scratch on a VGGFace2 subset (**8,631 identities**, 112×112 images) to classify faces. We then reuse the network without its classifier as a **face-embedding extractor** for open-set verification (**6,000 pairs**, **5,749 unseen identities**). The Kaggle score is `0.5·Accuracy + 0.5·(1 − EER)`.

Cutoffs: very low 80, low 88, medium 92, high 95.

| Run | Model | Loss | Val cls acc | Val EER | Val score | Kaggle |
|---|---|---|---|---|---|---|
| `run1_baseline_cnn5` | 5-layer CNN, 15.1M params | CE + label smoothing 0.1, 20 epochs | 79.89 | 5.30 | 87.30 | 87.33 |
| `run2_resnet34_arcface` | ResNet-34 layout, 26.0M params | ArcFace (s=32, m=0.4), 50 epochs, flip TTA; best epoch 47 | 93.21 | 3.06 | 95.08 | **95.74** |

The notebook currently configures **run two**, which cleared the high cutoff on the public leaderboard. It trained on PSC as Slurm job 47408105; the log is in `checkpoints/run2_resnet34_arcface/`. Run instructions are in [submission.md](submission.md).

## Run layout
Each run writes only to its own folder, `checkpoints/<run_name>/`, set by `config['run_name']`:

| File | Contents |
|---|---|
| `run_config.json` | The config the run was launched with |
| `best_combined.pth`, `best_cls.pth`, `best_ret.pth`, `last.pth` | Checkpoints. Each holds model, optimizer and scheduler state, metrics, epoch and config |
| `wandb_run_id.txt` | The wandb run id, so a resumed session logs to the same wandb run |
| `submission.csv`, `cls_submission.csv`, `ver_submission.csv` | Kaggle files (`submission.csv` is also copied to the repo root for submitting) |

With `config['resume'] = True`, re-running the notebook continues from that run's `last.pth`. This matters on PSC, where interactive sessions end after 8 hours.

Run one is archived in [checkpoints/run1_baseline_cnn5/](checkpoints/run1_baseline_cnn5/). Its [run_config.json](checkpoints/run1_baseline_cnn5/run_config.json) records the architecture, hyperparameters, metrics, wandb id (`dbo7trqy`) and source commit (`15cfe4c`).

## Training on PSC as a batch job
[psc/train.py](psc/train.py) is generated from the notebook's code cells by [psc/make_train_script.py](psc/make_train_script.py). After editing the notebook, regenerate it with `python psc/make_train_script.py`. It downloads the data to the job's `$LOCAL` disk, trains with resume, and writes the submission CSVs. It does not submit to Kaggle.

```bash
sbatch -t 0:30:00 --export=ALL,SMOKE=1 psc/run2.sbatch   # first: few-minute end-to-end check on PSC; log must end with SMOKE TEST PASSED
sbatch psc/run2.sbatch          # from /jet/home/<user>, next to the notebook, .env and utils/
squeue -u $USER                 # PD = waiting, R = running
tail -f run2-<jobid>.out        # log; training curves are on wandb
```
If the job stops before the last epoch, submit it again; it resumes from `last.pth`.

`*.pth` is git-ignored from run two onward. Run-two checkpoints with optimizer state are about 210 MB each, over GitHub's 100 MB limit.

## Key terms
| Term | Meaning here |
|---|---|
| **Classification** (closed set) | Predict which of the 8,631 training identities a face belongs to. Metric: accuracy |
| **Verification** (open set) | Decide whether two faces are the same person, including people never seen in training. Metric: EER |
| **Embedding / `feats`** | The 512-d BatchNorm-ed vector from the embedding head. Compared with **cosine similarity** |
| **EER** | Equal Error Rate: the threshold where false-accept rate = false-reject rate. Lower is better |
| **Residual block** | Two 3×3 convs plus an identity skip connection (writeup §4.4). Lets a 34-layer network train stably |
| **Strided conv** | Conv followed by downsampling, merged into one op (lecture 10). Used for every resolution drop |
| **ArcFace** | Normalizes embeddings and class weights, then requires the true class to win by an angular margin `m` (writeup §5.6.2). Produces compact, well-separated embeddings, which lowers EER |
| **Margin warmup** | `m` ramps from 0 to 0.4 over the first 5 epochs, so the margin doesn't block early learning |
| **LR warmup + cosine** | LR rises linearly for 2 epochs, then follows a half-cosine to 0. Stepped every batch |
| **Flip TTA** | At inference, embed the image and its mirror and sum the normalized embeddings |
| **Augmentation** | Random crop/scale, horizontal flip, color jitter, random erasing. No vertical flip (upside-down faces are unrealistic, per the writeup) |

## Where the model-building code lives (`F26 HW2P2 Student Starter Notebook_final.ipynb`)
| Section | What to look at |
|---|---|
| **Config** | `run_name`, architecture (`blocks`, `channels`, `embedding_dim`, `dropout`), ArcFace (`arcface_s`, `arcface_m`, `margin_warmup_epochs`), `epochs=50`, `warmup_epochs`, `resume` |
| **Data Augmentations** → `create_transforms` | Training augmentations, normalization to [-1, 1], then random erasing |
| **Datasets and DataLoaders** | `cls_data/{train,dev,test}` via `ImageDataset`. Verification pairs via `ImagePairDataset`. Test loaders never shuffle (submission order matters) |
| **Model Architecture** → `Network` | 3×3/s1 stem → 4 stages of `BasicBlock`s (112→56→28→14→7) → avg pool → Dropout → Linear → BN1d embedding → `NormLinear` cosine classifier. `FlipTTA` wraps the model for inference |
| **Loss / Optimizer / Scheduler** | `ArcFaceLoss`, SGD (Nesterov, no weight decay on BN/bias), `LambdaLR` warmup + cosine |
| **Metrics** → `verification_metrics` | ROC → EER (via `brentq`), AUC, best-threshold accuracy. Same formulas as the course's reference `metrics_calculator` |
| **Experiments** | Resume → set margin → train → validate classification and verification → combined score → per-run checkpoints |
| **Inference & Submission** | Loads the run's `best_combined.pth`, reports val EER with flip TTA, then the course utils write and submit the CSVs |

Credentials come from `.env`, which is git-ignored, and are parsed in the Kaggle-auth cell into `ENV`.
