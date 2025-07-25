# ABONN: 






<img src="https://i.postimg.cc/qRFzWcTt/mcts.png" alt="propose" width="450"/>

We propose **ABONN**, a framework that adapts Monte Carlo Tree Search (MCTS) techniques for neural network verification. 

The design of **ABONN** is based on two key ideas:
1. The branch-and-bound tree provides a natural, ___deterministic___ tree structure for sequentially searching for counterexamples, similar to exploring paths in a tree to find violations.
2. ABONN leverages recursive rollout and backtracking strategies, enabling effective balancing exploration and exploitation of the search space.

<details><summary> Installation </summary>
<p>
## 1.Configuration 
#### 1.1 Configuration for MiniConda
```
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh -b -u
source ~/miniconda3/bin/activate conda init bash

conda create -n py38 python=3.8 -y
conda activate py38
```
### 1.2 Configuration for Working environments

```
pip install -r requirements.txt
```

#### 1.3 Configuration for Gurobi

For Linux-based systems the installation steps are: 

Install Gurobi:
```
wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz
tar -xvf gurobi9.1.2_linux64.tar.gz
cd gurobi912/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cp libgurobi_c++.a ../../lib/
cd ../../
cp lib/libgurobi91.so /usr/local/lib -> (You may need to use sudo command for this)   
python3 setup.py install
cd ../../
```

```
export GUROBI_HOME="$HOME/opt/gurobi950/linux64"
export GRB_LICENSE_FILE="$HOME/gurobi.lic"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/$HOME/usr/local/lib:/usr/local/lib
```

Getting the free academic license To run GUROBI one also needs to get a free academic license. https://www.gurobi.com/documentation/9.5/quickstart_linux/retrieving_a_free_academic.html#subsection:academiclicense

a) Register using any academic email ID on the GUROBI website. 

b) Generate the license on https://portal.gurobi.com/iam/licenses/request/

Choose Named-user Academic

c)Use the command in the command prompt to generate the license.

(If not automatically done, place the license in one of the following locations “/opt/gurobi/gurobi.lic” or “$HOME/gurobi.lic”)

```
wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz 
tar -xvf gurobi9.1.2_linux64.tar.gz 
cd gurobi912/linux64/src/build 
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile make cp libgurobi_c++.a ../../lib/ 
cd ../../ cp lib/libgurobi91.so /usr/local/lib 
python3 setup.py install cd ../../
export GUROBI_HOME="$HOME/opt/gurobi950/linux64" 

```
- Apply for your gurobi.lic path
``` 
export GRB_LICENSE_FILE="$HOME/gurobi.lic" 
export PATH="${PATH}:${GUROBI_HOME}/bin" 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/$HOME/usr/local/lib:/usr/local/lib

conda config --add channels https://conda.anaconda.org/gurobi

conda install gurobi -y

```


-------------

## Part II: Verification


### II.1 Demonstration with a toy example

``` 
python mcts_demo.py 
```

As shown

```
===========Performing Traditional BaB ===========
cur_spec:  -2.6515151515151514 {}
cur_spec:  -2.5200000000000005 {(1, 1): -1}
cur_spec:  -2.6515151515151514 {(1, 1): 1}
cur_spec:  0.2333333333333334 {(1, 1): -1, (1, 0): -1}
cur_spec:  -0.75 {(1, 1): -1, (1, 0): 1}
cur_spec:  0.2333333333333334 {(1, 1): 1, (1, 0): -1}
cur_spec:  -2.033333333333333 {(1, 1): 1, (1, 0): 1}
find a counterexample 1.0 0.0 [-3.4]
Total analyzer calls:  7

```

<img src="https://i.postimg.cc/2S9zjSKG/mcts-Traditional-Bab.jpg" alt="BaB-baseline" width="450"/>

```
===========Performing ABONN ===========

cur_spec:  -2.6515151515151514 {}
cur_spec:  -2.5200000000000005 {(1, 1): -1}
cur_spec:  -2.6515151515151514 {(1, 1): 1}
cur_spec:  0.2333333333333334 {(1, 1): 1, (1, 0): -1}
cur_spec:  -2.033333333333333 {(1, 1): 1, (1, 0): 1}
find a counterexample 1.0 0.0 [-3.4]
Total analyzer calls:  5

```

<img src="https://i.postimg.cc/wjjqdm03/mcts-run.jpg" alt="mcts-baseline" width="450"/>




#### II.2 Running experiments

```
# e.g. To run CIFAR dataset with OVAL_WIDE model.
python verifier.py cifar10ovalwide 01
```

```
# e.g. To run MNIST dataset with L4 model.
python verifier.py mnistL4 01
```

## Part III Results 

Our presented results on the paper are all available at ./experiment

- All raw data executed results, please refer to ./experiment/csv

- All experimental figures utilized in our paper are summarized in ./experiment/ipynb


## License and Copyright

Licensed under the [Apache License](https://www.apache.org/licenses/LICENSE-2.0)
- Our implementation is built on top of 
    - [ERAN] https://github.com/eth-sri/eran
    - [IVAN] https://github.com/uiuc-arc/Incremental-DNN-Verification 
    
