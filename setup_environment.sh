## Copy and paste this text into the terminal to set up the environment

echo "Updating Conda"
conda update -n base conda

echo "Installing libmamba solver"
conda install -n base conda-libmamba-solver=22.12.0

echo "Setting libmamba solver"
conda config --set solver libmamba

echo "Creating new environment"
conda create --name myenv python=3.11 numpy pytorch torchaudio pytorch-cuda=11.7 torchvision matplotlib jupyter cudatoolkit scikit-learn -c pytorch -c nvidia
conda activate myenv

echo "Installing huggingface transformers"
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
cd ..

pip install psutil
pip install tensorboard
pip install transformers datasets accelerate nvidia-ml-py3 einops

cd transformers/examples/pytorch/language-modeling
pip install -r requirements.txt
cd ../../../..
