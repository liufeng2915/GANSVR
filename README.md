# **2D GANs Meet Unsupervised Single-View 3D Reconstruction**

European Conference on Computer Vision (ECCV 2022). [[Arxiv](), [PDF](http://cvlab.cse.msu.edu/pdfs/Liu_Liu_ECCV2022.pdf), [Supp](http://cvlab.cse.msu.edu/pdfs/Liu_Liu_ECCV2022_supp.pdf), [Project](http://cvlab.cse.msu.edu/project-gansvr.html)]

**[Feng Liu](http://cvlab.cse.msu.edu/pages/people.html), [Xiaoming Liu](http://cvlab.cse.msu.edu/pages/people.html)**

We propose a novel image-conditioned neural implicit field, which can leverage 2D supervisions from GAN-generated multi-view images and perform the single-view reconstruction of generic objects. Firstly, a novel offline StyleGAN-based generator is presented to generate plausible pseudo images with full control over the viewpoint. Then, we propose to utilize a neural implicit function, along with a differentiable renderer to learn 3D geometry from pseudo images with object masks and rough pose initializations. To further detect the unreliable supervisions, we introduce a novel uncertainty module to predict uncertainty maps, which remedy the negative effect of uncertain regions in pseudo images, leading to a better reconstruction performance. The effectiveness of our approach is demonstrated through superior single-view 3D reconstruction results of generic objects. 

## Prerequisites

The code is developed with

* Python 3.7
* Pytorch 18
* Cuda 11.1

## Training 

* Please follow [here](https://github.com/liuf1990/GANSVR/tree/main/code/data/README.md) to prepare the training data.

* Train the model:

  ```bash
  python training/exp_runner.py --conf ./confs/car.conf 

## Citation 

```bash
@inproceedings{liu2022gansvr,
title={2D GANs Meet Unsupervised Single-View 3D Reconstruction},
author={Liu, Feng and Liu, Xiaoming},
booktitle={ECCV},
year={2022}}
```

## Acknowledgments

Our implementation is heavily built upon [idr](https://github.com/lioryariv/idr). If you find our work useful, please also consider to cite this paper.

## License

[MIT License](LICENSE)

## Contact 

For questions feel free to post here or drop an email to - liufeng6@msu.edu

