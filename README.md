This repository containes our implementation for the approach presented in our paper: **<em>"See the silence: improving visual-only voice activity detection by optical flow and RGB fusion"</em>**. <br><br>
The official published paper is available here: https://link.springer.com/chapter/10.1007/978-3-030-87156-7_4 <br><br>
and an earlier version is also available here: https://github.com/ducspe/VVADpaper <br><br>
This program needs to be run twice: one time with RGB inputs, and second time with optical flow inputs. <br><br>
To create mean and standard deviation statistics, the **learn_trainingsubset_statistics.py** needs to be called for RGB and optical flow separately. The resulting .npy file will have to be named correspondingly. The file name string is used as a parameter in **vvad_train.py**, **vvad_test.py** and **vvad_fusion_test.py**, where 2 separate .npy files are necessary. <br><br>
**data_dda** is a folder with a very small example subset from the **TCD-TIMIT** preprocessed data. Full preprocessing code is available in a separate repository at: https://github.com/ducspe/TCD-TIMIT-Preprocessing After the preprocessing, the full dataset has to follow the structure of the **data_dda** folder in this repository. <br><br>
The train is started by running **vvad_train.py** <br><br>
You can then test without any fusion by running **vvad_test.py** <br><br>
The RGB and optical flow models need to be saved and fused with the help of **vvad_fusion_test.py** <br><br>
Code related to audio label inference is available in the **processing** folder/module <br><br>
All other helper utility functions are available in **utils.py**
