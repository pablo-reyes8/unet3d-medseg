import tarfile
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import os

!curl -L "https://drive.google.com/uc?export=download&id=1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C" -o "Task04_Hippocampus.tar"
!tar -xf Task04_Hippocampus.tar