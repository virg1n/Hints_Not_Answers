# Hints_Not_Answers

This repository accompanies the paper **"Making GenAI Socratic for Computational Thinking"**.

## Repository Structure

* **Dataset/**

  * Code for generating the dataset of buggy code, errors, and Socratic questions
  * The dataset itself (∼7,000 samples) used for fine-tuning and reward training

* **RLHF/**

  * Code to perform supervised fine-tuning (SFT.py) and reinforcement learning from human feedback (PPOrlhf.py)
  * Scripts and configurations for training the reward model (train_reward_model.py)

## Pre-trained Models

The fine-tuned models are available on Hugging Face:

[https://huggingface.co/Virg1n/hints\_not\_answers/](https://huggingface.co/Virg1n/hints_not_answers/)

## Instructions

To start training on your own:

1. Obtain our reward model from Hugging Face.
2. Run supervised fine-tuning:

   ```bash
   python sft.py
   ```
3. Launch the reward model server:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python reward_model_server.py
   ```
4. Run PPO RLHF training:

   ```bash
   python PPOrlhf.py
   ```


To train the reward model itself:

   ```bash
   python train_reward_model.py
   ```
