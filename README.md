# Light-Adapted Electroretinogram and Oscillatory Potentials (LEOPs) Dataset for Autism Spectrum Disorder and Typically Developing Individuals

Analysis scripts for the **LEOPs** dataset — light-adapted electroretinogram (ERG) and oscillatory potentials (OPs) waveforms from children and adolescents with Autism Spectrum Disorder (ASD), ASD with co-occurring ADHD (ASD+ADHD), and typically developing (TD) controls.

The dataset contains 5,309 ERG waveforms and 4,434 OPs waveforms from 253 participants recorded at two sites (Flinders University, Adelaide, Australia and University College London, UK) using the RETeval handheld device with skin electrodes.

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/MikhailKulyabin/LEOPs.git
cd LEOPs
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the LEOPs dataset from [Mendeley Data](https://data.mendeley.com/) and place its contents into the `dataset/` directory so the structure looks like:

```
LEOPs/
├── dataset/
│   ├── LEOPs_dataset.xlsx
│   ├── jsons/           # 253 JSON files (one per participant)
│   └── ImagesERG/       # 558 PNG eye images
├── rff_extract.py
├── generate_statistics.py
├── plot_waveforms.py
├── plot_ops_waveforms.py
├── plot_validation.py
├── requirements.txt
└── README.md
```

## Scripts

### `generate_statistics.py`

Compute summary statistics (mean +/- SEM) for ERG parameters grouped by protocol and diagnostic group. Outputs a CSV table to `tables/statistics.csv`.

```bash
python generate_statistics.py
```

### `plot_waveforms.py`

Plot mean +/- 1 SD ERG waveforms for all three protocols (9-step, 2-step, LA3), grouped by diagnostic category.

```bash
python plot_waveforms.py
```

Outputs: `figures/figure_9step.pdf`, `figures/figure_2step.pdf`, `figures/figure_LA3.pdf`

### `plot_ops_waveforms.py`

Plot mean +/- 1 SD oscillatory potentials waveforms for the 9-step and 2-step protocols.

```bash
python plot_ops_waveforms.py
```

Outputs: `figures/figure_ops_9step.pdf`, `figures/figure_ops_2step.pdf`

### `plot_validation.py`

Generate validation figures: photopic hill intensity-response curves and inter-ocular b-wave amplitude scatter plots.

```bash
python plot_validation.py
```

Outputs: `figures/figure_photopic_hill.pdf`, `figures/figure_interocular.pdf`


## Dataset structure

Each participant JSON file in `dataset/jsons/` has the following structure:

```
participant_id: string
demographics:
  category: int (0=Control, 1=ASD/ASD+ADHD)
  group: string (Control, ASD, ASD+ADHD)
  site: int (1=Adelaide, 2=London)
  sex: int (0=male, 1=female)
  ...
eye_images:
  right: filename | null
  left: filename | null
recordings: [
  wave_id, protocol, age, test_eye,
  stimulus: { flash_tds, flash_cd, frequency_hz },
  features: { a_time_ms, a_amp_uv, b_time_ms, b_amp_uv, ... },
  erg_waveform: { time_ms: [], amplitude_uv: [] },
  op_waveform: { time_ms: [], amplitude_uv: [] } | null
]
```

## Protocols

| Protocol | Flash strengths (Td.s) | Participants | Waveforms | Points/waveform |
|----------|----------------------|--------------|-----------|-----------------|
| 9-step   | 12, 21, 35, 70, 113, 178, 251, 356, 446 | 173 | 4,246 | 235 |
| 2-step   | 113, 446             | 61           | 415       | 430             |
| LA3      | 85                   | 217          | 648       | 235             |

