# Deep Learning Based Automatic Modulation Recognition: Models, Datasets, and Challenges
Source code for the paper "Deep Learning Based Automatic Modulation Recognition: Models, Datasets, and Challenges", which is published in Digital Signal Processing.

Representative and up-to-date models in the AMR field are implemented on four different datasets (RML2016.10a, RML2016.10b, RML2018.01a, HisarMod2019.1), providing a unified reference for interested researchers.

The article is available here:[Deep Learning Based Automatic Modulation Recognition: Models, Datasets, and Challenges](https://www.sciencedirect.com/science/article/pii/S1051200422002676?via%3Dihub)

If you have any question, please contact e-mail: zhangxx8023@gmail.com

# Abstract
Automatic modulation recognition (AMR) detects the modulation scheme of the received signals for further signal processing without needing prior information, and provides the essential function when such information is missing. Recent breakthroughs in deep learning (DL) have laid the foundation for developing high-performance DL-AMR approaches for communications systems. Comparing with traditional modulation detection methods, DL-AMR approaches have achieved promising performance including high recognition accuracy and low false alarms due to the strong feature extraction and classification abilities of deep neural networks. Despite the
promising potential, DL-AMR approaches also bring concerns to complexity and explainability, which affect the practical deployment in wireless communications systems. This paper aims
to present a review of the current DL-AMR research, with a focus on appropriate DL models and benchmark datasets. We further provide comprehensive experiments to compare the state of
the art models for single-input-single-output (SISO) systems from both accuracy and complexity perspectives, and propose to apply DL-AMR in the new multiple-input-multiple-output (MIMO)
scenario with precoding. Finally, existing challenges and possible future research directions are discussed.

# Content
## Experimental comparison for SISO system
### Accuracy

![Recognition accuracy comparison of the state-of-the-art models on (a) RML2016.10a, (b) RML2016.10b, (c) RML2018.01a, (d) HisarMod2019.1](https://user-images.githubusercontent.com/56213845/179749467-e08a9561-aec5-4741-8e72-1ae8b026638e.png)
**Fig.1** Recognition accuracy comparison of the state-of-the-art models on (a) RML2016.10a, (b) RML2016.10b, (c) RML2018.01a, (d) HisarMod2019.1.

### Parameter Comparison
**Table1** Model size and complexity comparison on the four datasets (A: RML2016.10a, B: RML2016.10b, C: RML2018.01a, D: HisarMod2019.1).
![1658233618155](https://user-images.githubusercontent.com/56213845/179749943-6b74c23d-dbff-4aef-9d4d-e0f3c8eb7993.png)

### Confusion matrix
![combine_revise2022512](https://user-images.githubusercontent.com/56213845/179750124-235922ad-f4e6-457f-937a-7b3466d921d9.png)
**Fig.2** Confusion matrices. A, B and C represent the confusion matrices obtained on the RML2016.10a, RML2016.10b, and RML2018.01a, respectively. The numerical indexes 1 - 14 denote CNN1, CNN2, MCNET, IC-AMCNET, ResNet, DenseNet, GRU, LSTM, DAE, MCLDNN, CLDNN, CLDNN2, CGDNet, PET-CGDNN.

# Dataset

**Table2** Main AMR open datasets for SISO systems.
![1658233963147](https://user-images.githubusercontent.com/56213845/179750964-f49c2657-3348-48b2-86bc-dd3855b56378.png)

| Dataset | Link |Notes |
| :-----:| :----: | :----: |
| [RML2016.10a, RML2016.10b](https://pubs.gnuradio.org/index.php/grcon/article/view/11), [RML2018.01a](https://ieeexplore.ieee.org/abstract/document/8267032)| [RML](http://radioml.com) | If RML2018 dataset is too large, you can use SubsampleRML2018.py to sample the dataset to get a partial dataset for experimentation. |
| [HisarMod2019.1](https://ieeexplore.ieee.org/abstract/document/9128408) | [HisarMod](http://dx.doi.org/10.21227/8k12-2g70) | In our experiments, the dataset was converted from a .CSV file to a .MAT file, which can be found in [Link](https://pan.baidu.com/s/1ChAMTrTnhgaIBmp9NmFG-Q?pwd=s54g).|


# Related Papers
| Model | Paper name | Publication year |
| :-----:| :----: | :----: |
| CNN1| [Convolutional Radio Modulation Recognition Networks](https://link.springer.com/chapter/10.1007/978-3-319-44188-7_16) | 2016  |
|CNN2| [Robust and Fast Automatic Modulation Classification with CNN under Multipath Fading Channels](https://ieeexplore.ieee.org/abstract/document/9128408)  | 2020   |
|MCNET| [MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification](https://ieeexplore.ieee.org/abstract/document/8963964)  |  2020  |
|IC-AMCNET| [CNN-Based Automatic Modulation Classification for Beyond 5G Communications](https://ieeexplore.ieee.org/abstract/document/8977561) | 2020   |
|ResNet|[Deep neural network architectures for modulation classification](https://ieeexplore.ieee.org/abstract/document/8335483)   |  2017  |
|DenseNet|[Deep neural network architectures for modulation classification](https://ieeexplore.ieee.org/abstract/document/8335483)    |  2017  |
|GRU| [Automatic Modulation Classification using Recurrent Neural Networks](https://ieeexplore.ieee.org/abstract/document/8322633)  | 2017   |
|LSTM|[Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors](https://ieeexplore.ieee.org/abstract/document/8357902)  |   2018 |
|DAE|[Real-Time Radio Technology and Modulation Classification via an LSTM Auto-Encoder](https://ieeexplore.ieee.org/abstract/document/9487492)| 2022   |
|MCLDNN| [A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition](https://ieeexplore.ieee.org/abstract/document/9106397)  | 2020   |
|CLDNN|[Deep Architectures for Modulation Recognition](https://ieeexplore.ieee.org/abstract/document/7920754) |2017    |
|CLDNN2|[Deep neural network architectures for modulation classification](https://ieeexplore.ieee.org/abstract/document/8335483)    |  2017  |
|CGDNet| [CGDNet: Efficient Hybrid Deep Learning Model for Robust Automatic Modulation Recognition](https://ieeexplore.ieee.org/abstract/document/9349627)  | 2021   |
|PET-CGDNN|[An Efficient Deep Learning Model for Automatic Modulation Recognition Based on Parameter Estimation and Transformation](https://ieeexplore.ieee.org/abstract/document/9507514)|2021    |
|1DCNN-PF|[Automatic Modulation Classification Using Parallel Fusion of Convolutional Neural Networks](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Automatic+Modulation+Classification+Using+Parallel+Fusion+of+Convolutional+Neural+Networks&btnG=)   | 2019   |

# Environment
These models are implemented in Keras, and the environment setting is:
* Python 3.6.10
* TensorFlow-gpu 1.14.0
* Keras-gpu 2.2.4

# Remarks
You will need to download the appropriate dataset and change the flie path to the corresponding dataset in your code. There is no guarantee that the code can run sucessfully under other environmental configurations, but there may be performance differences due to different hardware conditions.

About DAE: In the author's open source code, decoder uses the TimeDistributed layer. In our implementation, decoder unfolds the data and uses a fully connected layer to reconstruct the input, so the difference is described here. [(Source code for DAE)](https://github.com/WuLoli/LSTMDAE)  

# Acknowledgement
Our code is partly based on [leena201818](https://github.com/leena201818). Thanks [leena201818](https://github.com/leena201818) and [wzjialang](https://github.com/wzjialang/MCLDNN#introduction) for their great work!

# Citation
Please cite the literature we refer to if they are helpful to your work.
If our work is helpful to your research, please cite:

    @article{ZHANG2022103650,
        title={Deep Learning Based Automatic Modulation Recognition: Models, Datasets, and Challenges},
        author={Fuxin Zhang and Chunbo Luo and Jialang Xu and Yang Luo and FuChun Zheng},
        journal={Digital Signal Processing},
        year={2022},
        doi = {https://doi.org/10.1016/j.dsp.2022.103650}
    }
