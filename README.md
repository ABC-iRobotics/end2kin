# end2kin

## Overview

Minimally Invasive Surgery (MIS) and Robot-Assisted MIS (RAMIS) can help to improve the patient outcome
through small skin incision, less blood loss, smaller scars and quicker recovery time. However, for surgeons to master, both MIS
and RAMIS require extensive training. Autonomous surgical skill
assessment can provide feedback to the surgeon, and can help
with personalized training as well. Kinematic data is proven to
be an effective tool for surgical skill classification, nevertheless,
2D image data-based surgical skill assessment is still an open
challenge. If a motion data—that is strongly correlated with
the kinematic data—based on 2D images can be calculated,
it can significantly improve the accuracy of image-based skill
assessment solutions. In this work, a surgical tool pose estimation
technique was introduced for the da Vinci Surgical System’s
articulated tools, targeting autonomous technical skill assessment
based on 2D endoscopic images. The pose estimation was done
by shape features of the surgical tools, optical flow and iterative
perspective n point transformation; the method does not require
markers, kinematic data or the CAD model of the tool. The
introduced technique was validated on the Synthetic MICCAI
dataset, where it resulted 4.22, 4.23 and 3.95 mm mean absolute
translational error along x, y, z axes, respectively, based on 8
videos, which contained 1906 frames altogether. The estimated
rotation was not evaluated in the later experiments due to its
inaccuracy. The motion smoothness features (log of Dimensionless
Jerk and Spectral Arc Length) did not show significant differ-
ences between the generated and the original motion trajectories
based on Wilcoxon signed rank tests. To prove the applicability
of surgical skill assessment, the experienced noise added to
JIGSAWS kinematic data, and skill classification was done with
time series forest classifier. The results were 76.66%, 75% and
89% mean accuracy for suturing, needle-passing and knot-tying,
respectively, where suturing accuracy outperforms the state of
the art in 2D image-based solutions on the JIGSAWS dataset.

## Data

To test and validate the autonomous joint detection and pose estimation, a RAMIS video dataset was necessary, which annotated not only with the segmentation ground truth, but the robot kinematic data as well.
Synthetic MICCAI dataset can be found here: https://www.ucl.ac.uk/interventional-surgical-sciences/weiss-open-research/weiss-open-data-server/ex-vivo-dvrk-segmentation-dataset-kinematic-data

## Dependencies

To install these dependencies you can use requirements.txt file.

## Usage

Tool pose estimation:

    python main.py
    
## Citation

If you find this work useful for your publications, please consider citing:

    @inproceedings{elek2022towards,
      title={Towards Autonomous Endoscopic Image-Based Surgical Skill Assessment: Articulated Tool Pose Estimation},
      author={Elek, Ren{\'a}ta Nagyn{\'e} and Haidegger, Tam{\'a}s},
      booktitle={2022 IEEE 10th Jubilee International Conference on Computational Cybernetics and Cyber-Medical Systems (ICCC)},
      pages={000035--000042},
      year={2022},
      organization={IEEE}
    }
