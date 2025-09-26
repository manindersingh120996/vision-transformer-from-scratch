# Vision Transformer (ViT) from Scratch in PyTorch
[Paper Link�](https://arxiv.org/abs/2010.11929)

Hey everyone! 

I wanted to figure out how Vision Transformers (ViTs) really work, so I decided to build one from the ground up.

This repo is basically my step-by-step adventure in turning the famous "An Image is Worth 16x16 Words" paper into actual, working PyTorch code. I built everything piece by piece, from the image patching to the final training loop. If you're curious about ViTs, hopefully this simple breakdown helps!
<img width="979" height="601" alt="screenshot of architecture from Original an Image is worth 16 X 16 words" src="https://github.com/user-attachments/assets/1986a0ba-28cd-4dde-af92-443c81bb95c6" />

### What's the Big Idea?
The main idea behind ViT is pretty cool. It treats little patches of an image like words in a sentence. It then feeds these "image words" into a Transformer—the same kind of model that's great at understanding language—to figure out what's in the picture.

The whole process looks something like this:

- Chop up the image into a grid of smaller patches.

- Turn patches into embeddings (a list of numbers the model can understand).

- Add position info so the model knows where each patch came from.

- Feed it through the Transformer Encoder, where the model finds patterns and relationships between the patches.

- Use a final classifier head to make a prediction.

<img width="639" height="636" alt="sample cropped up image in patches" src="https://github.com/user-attachments/assets/4a463a32-d5d8-4927-946c-19144d941e6d" />

### How the Project is Organized
Here’s how I’ve laid everything out to keep it from getting messy:

``` python
├── configs/
│   └── config.yaml          # Hydra configuration file for all hyperparameters
├── dataset_path/
│   └── ...                 
├── outputs/
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/        # Outputs from a single run (logs, model, etc.)
├── multirun/
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/        # Parent folder for multiple experiment runs
│           ├── 0/
│           └── 1/
├── src/
│   ├── __init__.py
│   ├── model.py             # All nn.Module classes for the ViT architecture
│   ├── dataset.py           # Your custom PyTorch Dataset and transforms
├── .gitignore
├── train.py                 # The main script to start training
└── inference.py               # The script to run inference with a trained model
```
### How to Get It Running
<img width="1464" height="491" alt="Training result" src="https://github.com/user-attachments/assets/78cc7496-9a9f-471f-adaa-8d34c0f72adc" />
Sample Training result on Tom and jerry dataset for 100 epochs, logs can be found in `outputs\2025-09-25\21-04-19\`

<br/>
<br/>
*Before starting it if you want to use the dataset of tom and jerry then plese use the script in top 4-5 cells in file "base_experimentation.ipynb". 
Also, if after downloading from kaggle, the data is not showing in you local repo then you can find it in .cache folder of you system, depending upon OS.*
<br/>
<br/>
Getting started is pretty straightforward.

1. Set Things Up
First, you'll want to grab the code and install the packages it needs.

```Bash

git clone "https://github.com/manindersingh120996/vision-transformer-from-scratch.git"
cd "vision-transformer-from-scratch"
pip install -r requirements.txt
```
2. Tweak the Settings
The best part is that you can change almost anything without digging through the Python code. All the settings live in the `configs/config.yaml` file. Want to try a smaller model or a different learning rate? Just edit the text in that file.

```YAML

# configs/config.yaml
model:
  embedding_dim: 192   # Try making the model smaller or bigger
  num_layers: 6        # Use more or fewer Transformer blocks

training:
  epochs: 50           # Train for more or fewer epochs
  lr: 0.001            # Experiment with the learning rate!
```
3. Start Training!
Running it is just a couple of terminal commands.

To run a single training session using the settings in config.yaml:

```Bash

python train.py
```

To quickly try a different setting without editing the file:

```Bash

python train.py training.lr=0.005 model.num_layers=8
```
To run a bunch of experiments at once (this is super useful!):

```Bash

python train.py --multirun model.num_layers=4,6,8 training.lr=0.001,0.0005
```
This command will automatically run 6 different experiments for you, and Hydra will keep all the results in a nice, organized folder.

## Making Predictions with a Trained Model
After you've trained a model (or a few!), you can use predict.py to test it on new images. It’s a neat little script that finds all your saved models, lets you pick the one you want, and then makes a prediction.

Just run the script and follow the prompts:

```Bash

python inference.py
```

## How I Designed This (My Learning Journey)
My main goal for this project was to turn the concepts from the ViT paper into clean, modular, and understandable code.

- I broke down the entire architecture into small, self-contained nn.Module classes in src/model.py, like PatchEmbedding, MultiHeadAttention, and EncoderBlock. Then, I assembled them piece by piece, just like building with LEGOs.

- I used Hydra for configuration because I quickly realized how messy it is to have numbers like learning rates and model sizes hard-coded. This approach keeps the code clean and makes it super easy to experiment.

- The training loop is designed to save not just the model, but all the artifacts needed to reproduce or analyze the run: the config, a CSV of the metrics, plots of the training curves, and TensorBoard logs.

## Transfer Learning with this Model
The model you train on a custom dataset (like dataset used in original paper like `JFT-300M or ImageNet`) can serve as a base model for other similar tasks.
The process of taking a trained model and further training it on a new, related task is called transfer learning or fine-tuning. This is a powerful technique to get good performance on a new task without having to train from scratch again.

# This is the core thing which made ViT so much successfull and it is also it's downside that for a smaller dataset, conventional CNN architecture will out perform ViT because of in-built inductive Biasnes in CNNs.

### What's Next? (Future Plans 🚀)
I'm excited to keep building on this project. Here are a few things I have in mind for the future:

- TensorBoard Integration: Fully integrate TensorBoard for live, interactive visualization of all training metrics.

- DDP Training: Implementing Distributed Data Parallel (DDP) to allow for much faster training across multiple GPUs.

- Web Frontend: Build a simple frontend using Streamlit or Flask to provide a user-friendly interface for both training and inference.

- Cloud Deployment: Package the project with Docker and create a pipeline to train and deploy the model on a cloud platform like AWS, Azure, or GCP.
