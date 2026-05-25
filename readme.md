# Uncertainty-guided Morphometric-enhanced Diversity Modeling in Source-free Active Domain Adaptive Cross-center Tumor Segmentation
Official PyTorch implementation of **UMDM-SFADA** for cross-center medical image segmentation.
<div align="center">
  <img src="code/framework.png" width="85%">
  <p><b>图 1.</b> UMDM-SFADA 整体框架图</p>
</div>

##📦 Dataset Preparation
```
data/
└── YOUR_DATASET/
    │
    ├── 📄 Index files (.txt)
    │   ├── trainlist.txt                     # 3D training samples
    │   ├── train_slices.txt                  # 2D training slices
    │   ├── vallist.txt                       # Validation samples
    │   ├── testlist.txt                      # Testing samples
    │   ├── stage1_{active_method}.txt        # Stage-1: actively selected samples
    │   ├── all_{active_method}.txt           # Stage-2: all 3D samples
    │   └── all_slice_{active_method}.txt     # Stage-2: all 2D slices
    │
    ├── 📦 3D volumes (.h5)
    │   ├── case001.h5
    │   ├── case002.h5
    │   └── ...
    │
    └── 📂 slices/                            # 2D slices extracted from 3D volumes
        ├── case001_slice_0.h5
        ├── case001_slice_1.h5
        └── ...
```
### 📥 Preprocessed Data Download

To facilitate reproducibility, we provide our **preprocessed datasets** (including all `.h5` files and index `.txt` files) via Baidu Netdisk:

🔗 **Baidu Netdisk Link**: [https://pan.baidu.com/s/12-syXo7j-8mjQTEdQyYDcQ](https://pan.baidu.com/s/12-syXo7j-8mjQTEdQyYDcQ)  
🔑 **Extraction Code**: `d97y`

After downloading, extract the files and update the `base_dir` argument in the training scripts to point to your local dataset path.

> 📌 If you wish to use your own dataset, please organize it following the structure above.
### 📚 Dataset Citation

If you use the preprocessed datasets provided in this repository, please cite the corresponding original papers in addition to our work:

**NPC-GTV Segmentation**
```bibtex
@article{li2025dataset,
  title={A dataset of primary nasopharyngeal carcinoma MRI with multi-modalities segmentation},
  author={Li, Yin and Chen, Qi and Li, Meige and Si, Liping and Guo, Yingwei and Xiong, Yu and Wang, Qixing and Qin, Yang and Xu, Ling and Smagt, Patrick van der and others},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={1450},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
@article{wang2024dual,
  title={Dual-reference source-free active domain adaptation for nasopharyngeal carcinoma tumor segmentation across multiple hospitals},
  author={Wang, Hongqiu and Chen, Jian and Zhang, Shichen and He, Yuan and Xu, Jinfeng and Wu, Mengwan and He, Jinlan and Liao, Wenjun and Luo, Xiangde},
  journal={IEEE Transactions on Medical Imaging},
  year={2024},
  volume={43},
  number={12},
  pages={4078-4090},
  publisher={IEEE}
}
@article{luo2023deep,
  title={Deep learning-based accurate delineation of primary gross tumor volume of nasopharyngeal carcinoma on heterogeneous magnetic resonance imaging: A large-scale and multi-center study},
  author={Luo, Xiangde and Liao, Wenjun and He, Yuan and Tang, Fan and Wu, Mengwan and Shen, Yuanyuan and Huang, Hui and Song, Tao and Li, Kang and Zhang, Shichuan and others},
  journal={Radiotherapy and Oncology},
  volume={180},
  pages={109480},
  year={2023},
  publisher={Elsevier}
}
@article{luo2025generalizable,
  title={Generalizable Magnetic Resonance Imaging-based Nasopharyngeal Carcinoma Delineation: Bridging Gaps Across Multiple Centers and Raters With Active Learning},
  author={Luo, Xiangde and Wang, Hongqiu and Xu, Jinfeng and Li, Lu and Zhao, Yue and He, Yuan and Huang, Hui and Xiao, Jianghong and Song, Tao and Zhang, Shichuan and others},
  journal={International Journal of Radiation Oncology* Biology* Physics},
  volume={121},
  number={5},
  pages={1384--1393},
  year={2025},
  publisher={Elsevier}
}
```
**BC Segmentation**
```bibtex
@article{garrucho2025large,
  title={A large-scale multicenter breast cancer DCE-MRI benchmark dataset with expert segmentations},
  author={Garrucho, Lidia and Kushibar, Kaisar and Reidel, Claire-Anne and Joshi, Smriti and Osuala, Richard and Tsirikoglou, Apostolia and Bobowicz, Maciej and Del Riego, Javier and Catanese, Alessandro and Gwo{\'z}dziewicz, Katarzyna and others},
  journal={Scientific data},
  volume={12},
  number={1},
  pages={453},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
