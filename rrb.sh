python3 rotate.py r ./pict/a ./pict/b
cd ~/Repos/DeblurGANv2

# python3 predict.py ../Diploma-1/pict/b/* --out_dir ../Diploma-1/pict/c/
python3 predict.py '../Diploma-1/pict/a/*' --weights_path=./pretrained_models/fpn_inception.h5 --out_dir=../Diploma-1/pict/d/base/

python3 predict.py '../Diploma-1/pict/b/*' --weights_path=./pretrained_models/best_fpn_l3.h5  --out_dir=../Diploma-1/pict/c/l03/
python3 predict.py '../Diploma-1/pict/b/*' --weights_path=./pretrained_models/best_fpn_l5.h5  --out_dir=../Diploma-1/pict/c/l05/
python3 predict.py '../Diploma-1/pict/b/*' --weights_path=./pretrained_models/best_fpn_l7.h5  --out_dir=../Diploma-1/pict/c/l07/
python3 predict.py '../Diploma-1/pict/b/*' --weights_path=./pretrained_models/best_fpn_l9.h5  --out_dir=../Diploma-1/pict/c/l09/
python3 predict.py '../Diploma-1/pict/b/*' --weights_path=./pretrained_models/best_fpn_l11.h5 --out_dir=../Diploma-1/pict/c/l11/

python3 predict.py '../Diploma-1/pict/c/l03/*' --weights_path=./pretrained_models/fpn_inception.h5 --out_dir=../Diploma-1/pict/c/l03/da/
python3 predict.py '../Diploma-1/pict/c/l05/*' --weights_path=./pretrained_models/fpn_inception.h5 --out_dir=../Diploma-1/pict/c/l05/da/
python3 predict.py '../Diploma-1/pict/c/l07/*' --weights_path=./pretrained_models/fpn_inception.h5 --out_dir=../Diploma-1/pict/c/l07/da/
python3 predict.py '../Diploma-1/pict/c/l09/*' --weights_path=./pretrained_models/fpn_inception.h5 --out_dir=../Diploma-1/pict/c/l09/da/
python3 predict.py '../Diploma-1/pict/c/l11/*' --weights_path=./pretrained_models/fpn_inception.h5 --out_dir=../Diploma-1/pict/c/l11/da/

cd ~/Repos/Diploma-1

python3 rotate.py rb ./pict/c/l03/da ./pict/d/l03
python3 rotate.py rb ./pict/c/l05/da ./pict/d/l05
python3 rotate.py rb ./pict/c/l07/da ./pict/d/l07
python3 rotate.py rb ./pict/c/l09/da ./pict/d/l09
python3 rotate.py rb ./pict/c/l11/da ./pict/d/l11
