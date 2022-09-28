# cDCGAN(Conditional DCGAN)
A conditional generative adversarial network (CGAN) is a type of GAN that also takes advantage of labels during the training process.

* **Generator**: Given a label and uniform random variable array as input, and this network builds a mapping function from prior noise to our data space.

* **Discriminator**: Given batches of labeled data containing observations from both the training data and generated data from the generator, this network attempts return a single scalar representing the probability that x came form training data rather than generator distribution.

![image](https://user-images.githubusercontent.com/47561760/192095950-22efa2bf-ab8b-43d0-a3b7-36772ef3abc1.png)

The goal of the generator is to fool the discriminator, so the generative neural network is trained to maximise the final classification error (between true and generated data)

The goal of the discriminator is to detect fake generated data, so the discriminative neural network is trained to minimise the final classification error

[Paper](https://arxiv.org/abs/1411.1784)

# Dataset
The MNIST database of handwritten digits has a training set of 60,000 examples and a test set of 10,000 samples.
I used pytorch datasets for downloading dataset : 
```
train_dataset = datasets.MNIST('mnist/', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('mnist/', train=False, download=True, transform=transform)
```

# Model

Below is a model architecture diagram for a Conditional DCGAN. Note that the high-level architecture is essentially the same as in the cGAN, except the Generator and Discriminator contain additional layers, such as Convolutions and Transposed Convolutions.

![image](https://user-images.githubusercontent.com/47561760/192096043-2988611f-a0d8-43b1-be21-0af33136163d.png)

# Loss Function
![image](https://user-images.githubusercontent.com/47561760/192096070-c79cfb54-a875-498b-9118-64305973e8d8.png)


# Train
Trainer class Does the main part of code which is training model, plot the training process and save model each n epochs.

I Defined `Adam` Optimizer with learning rate 0.0002.

Each generative model training step occurse in `train_generator` function, descriminator model training step in `train_descriminator` and whole trining process in 
`train` function.

## Some Configurations
 
*   You can set epoch size : `EPOCHS` and batch size : `BATCH_SIZE`.
*   Set `device` that you want to train model on it : `device`(default runs on cuda if it's available)
*   You can set one of three `verboses` that prints info you want => 0 == nothing || 1 == model architecture || 2 == print optimizer || 3 == model parameters size.
*   Each time you train model weights and plot(if `save_plots` == True) will be saved in `save_dir`.
*   You can find a `configs` file in `save_dir` that contains some information about run. 
*   You can choose Optimizer: `OPTIMIZER` 

# Result

![c-dcgan](https://user-images.githubusercontent.com/47561760/192878304-8893a17a-bdc5-4c43-8d18-586d88249912.png)
