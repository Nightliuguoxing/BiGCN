#! /bin/bash
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install torch_geometric
unzip -d ./data/Weibo ./data/Weibo/weibotree.txt.zip
mkdir ./data/Weibograph
python ./Process/getWeibograph.py
python ./model/Weibo/BiGCN_Weibo.py 100
