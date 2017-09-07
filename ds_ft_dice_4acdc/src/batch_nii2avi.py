import os
from nii2avi import save_nii2avi

nii_folder = '/home/xinyang/project_xy/xy_acdc_2017/dataset/common/acdc_map'
img_n = 100

for k in range(img_n):
    print 'converting No. %d volume...' % (k)
    nii_path = os.path.join(nii_folder, ('pat_' + str(100+k) + '.0_seg.nii.gz'))
    video_path = os.path.join(nii_folder, ('pat_' + str(100+k) + '.mp4'))
    save_nii2avi(niipath=nii_path, Save_file_name=video_path, Time_Loop=40, Avi_rate=5, Angle=20)
