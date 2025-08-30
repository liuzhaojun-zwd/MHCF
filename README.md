# Heat Conduction Operator for Alleviating Modal Bias in Multi-modal Object Re-identification

![](.\picture\MHCF.jpg)

------

**M**ulti-modal **H**eat **C**ycle **F**usion (**MHCF**) is inspired by the physical mechanism of heat conduction. We design a cross-modal heat conduction operator to simulate the “information heat flow” that circulates and reaches dynamic equilibrium across different modalities. Extensive experiments conducted on several challenging multi-modal Re-ID benchmarks demonstrate that our approach significantly outperforms existing methods. By leveraging this progressive heat circulation mechanism, **MHCF** effectively mitigates the heterogeneity gap among modalities and achieves superior performance in terms of modality balance and robustness.

------



## **Results**

#### Multi-Modal Person ReID [RGBNT201 & Market-MM]

![](.\picture\person_result.png)

#### Multi-Modal Vehicle ReID [RGBNT100 & MSVR310]

![](.\picture\vehicle_result.png)

------



## **Visualizations**

![](.\picture\cam.jpg)

------



## **Reproduction**

### Datasets

- [**RGBNT201**](https://pan.baidu.com/s/1i8Yrd4fn2M9l67yyQV0dew?pwd=y4e5)
- [**Market-MM**](https://pan.baidu.com/s/1L1_RNbYghCiLpaZKRY_3HQ?pwd=nm6y)
- [**MSVR310**](https://pan.baidu.com/s/1hLoda7hkcDQzTEzro4OXVA?pwd=waxr)
- [**RGBNT100**](https://pan.baidu.com/s/1Ery15UYDHn4bVK67zA6EGQ?pwd=bm9z) 



### Training

```
conda create -n MHCF python=3.10.14 -y 
conda activate MHCF
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
cd (your_path)
pip install -r requirements.txt
python train.py --config_file ./configs/RGBNT201/MHCF.yml
```



### Testing

```
python test.py --config_file ./configs/RGBNT201/MHCF.yml
```



#### model pth

|  Dataset  | mAP  | R-1  |                            Model                             |
| :-------: | :--: | :--: | :----------------------------------------------------------: |
| RGBNT201  | 81.7 | 84.2 | [<u>**model**</u>](https://pan.baidu.com/s/1lhtZJEGhP2azrYmpXs5C3g?pwd=kjx2) |
| Market-MM | 85.2 | 94.3 | [**<u>model</u>**](https://pan.baidu.com/s/1nqReOheFgv-m5eTJl5kClA?pwd=uayf) |
| RGBNT100  | 87.2 | 97.1 | <u>**[model](https://pan.baidu.com/s/10RytRH5XE9K4or7K0j2T5g?pwd=4js3)**</u> |
|  MSVR310  | 59.3 | 68.1 | [<u>**model**</u>](https://pan.baidu.com/s/1t1a_HMnysdWO45jguFtmOg?pwd=ezyt) |



### Note

This repository is based on [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID). 
