# mano_pybullet_22DOF

[MANO](http://mano.is.tue.mpg.de/)-based hand models for the [PyBullet](https://pybullet.org/wordpress/) simulator.

The package is modified from [mano_pybullet](https://github.com/ikalevatykh/mano_pybullet).

## Update
We update a hand model with a degree of freedom (Dof) of 22. The reconstructed rotation axes remain consistent with the `AxisLayerFK` defined in [manotorch](https://github.com/lixiny/manotorch). In addition, we have provided the corresponding URDF and mesh files under `urdf` folder.
## Install

### From source code

```
git clone https://github.com/KailinLi/mano_pybullet.git
cd mano_pybullet
pip install -e .
```
Make sure you also install the following dependencies:
```
urdfpy
manotorch (https://github.com/lixiny/manotorch)
```


## Download MANO models

- Register at the [MANO website](http://mano.is.tue.mpg.de/) and download the models.
- Unzip the file mano_v1_2.zip, and put the folder `mano_v1_2` into the `assets` folder.


### Run tests

```
python demo.py
```
You should see the following demostation:

![demo](media/demo.gif)


## Citation
If you find `mano_pybullet` and `AxisLayerFK` useful in your research, please cite the repository using the following BibTeX entry.
```
@Misc{kalevatykh2020mano_pybullet,
  author =       {Kalevatykh, Igor et al.},
  title =        {mano_pybullet - porting the MANO hand model to the PyBullet simulator},
  howpublished = {Github},
  year =         {2020},
  url =          {https://github.com/ikalevatykh/mano_pybullet}
}

@inproceedings{yang2021cpf,
  title={CPF: Learning a contact potential field to model the hand-object interaction},
  author={Yang, Lixin and Zhan, Xinyu and Li, Kailin and Xu, Wenqiang and Li, Jiefeng and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11097--11106},
  year={2021}
}
```
## License
mano_pybullet is released under the [GPLv3](https://github.com/ikalevatykh/mano_pybullet/blob/master/LICENSE).