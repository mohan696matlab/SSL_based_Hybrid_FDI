# Prior Knowledge-Infused Self-Supervised Learning and Explainable AI for Fault Detection and Isolation in PEM Electrolyzers

## Overview

Repository containing the codes to reproduce the results presented in the  [research paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4641395)

## Files and Descriptions

- `sensor_data.csv, Electrolyzer_faults.csv`: [Fault dataset generated from the experimental setup one containing raw sensor measurements and one containing residual signals]
- `part_1_Hybrid_FDI_SSL_LFT_BG.ipynb`: [The proposed algorithm is demonstrated on the Electrolyzer dataset]
- `part_2_Ablation_Study.ipynb`: [Effect of various parameters on the proposed algorithm]
- `part_3_XAI_Occlusion.ipynb`: [BG-XAI method to explain the decision given by the deep learning model]
- `helper_function.py`: [Contains a bunch of helper functions]
- `StateOfArt1.ipynb, StateOfArt2.ipynb, StateOfArt3.ipynb`: [Comparision results with other state of the art methods]


## Installation

1. Clone the repository:

    ```bash
    git clone [https://github.com/your-username/your-repo.git](https://github.com/mohan696matlab/SSL_based_Hybrid_FDI.git)
    cd your-repo
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage
All the files are named from part_1 to part_3 and then SOTA (State of the Art) results are shown.
