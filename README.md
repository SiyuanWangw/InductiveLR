# InductiveLR

The source code of Paper "Query Structure Modeling for Inductive Logical Reasoning Over Knowledge Graphs".

If you find this paper useful, please cite this paper:
```
@misc{wang2023query,
      title={Query Structure Modeling for Inductive Logical Reasoning Over Knowledge Graphs}, 
      author={Siyuan Wang and Zhongyu Wei and Meng Han and Zhihao Fan and Haijun Shan and Qi Zhang and Xuanjing Huang},
      year={2023},
      eprint={2305.13585},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Setting up
1. To set up the environment, please install the packages in the `requirements.txt`.
```bash
pip install -r requirements.txt
```

2. You can first download the raw data to `Data` directory and run scripts in `DataProcessing` to get data for experiments.
 * For inductive datasets, please refer to Project [QE-TEMP](https://github.com/zhiweihu1103/QE-TEMP) to download the dataset [Ind-FB15k-237-V2](https://drive.google.com/drive/folders/1nrtn6ZhT2YZAW_313CRUJXCcG4kyRptv?usp=share_link) and dataset [Ind-NELL-V3](https://drive.google.com/drive/folders/1RDq1r5I29kmlGGyfukTgWEGe01OdZH4z?usp=share_link). Then 
 * For transductive datasets, please refer to Project [KGReasoning](https://github.com/snap-stanford/KGReasoning) to download the dataset FB15k, FB15k-237 and NELL995.
Then you can run the following scripts:
```bash
cd DataProcessing
python load_reasoning_data.py
python load_reasoning_data_ind.py
```
