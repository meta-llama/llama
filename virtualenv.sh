rm -rf llama_env
python3 -m venv llama_env
source llama_env/bin/activate

python -m pip install --upgrade pip

python -m pip install wheel
python setup.py bdist_wheel

pip install -r requirements.txt
pip install -e .

python -m pip install gradio

# run webapp.sh

