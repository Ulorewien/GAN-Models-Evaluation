import math
import tqdm
import numpy as np
import torch
from torch import optim
import torchvision.utils as torch_utils
from generator import Generator
from discriminator import Discriminator
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import gradient_penalty, generate_examples, check_loader, get_loader

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
dataset_dir = "/kaggle/input/women-clothes"
start_train_img_size = 4
lr = 1e-3
batch_sizes = [256,256,128,64,32,16]
channels_img = 3
z_dim = 512
w_dim = 512
in_channels = 512
lambda_gp = 10
progressive_epochs = [30] * len(batch_sizes)
gen_save_path = "stylegan_gen.pth"
disc_save_path = "stylegan_disc.pth"
training_plot_save_path = "train_plot.png"
animation_save_path = "animation.mp4"

check_loader() 

# Define models and optimizers
gen_net = Generator(z_dim, w_dim, in_channels, channels_img).to(device)
disc_net = Discriminator(in_channels, channels_img).to(device)
gen_optimizer = optim.Adam([{"params": [param for name, param in gen_net.named_parameters() if "map" not in name]},
                     {"params": gen_net.map.parameters(), "lr": 1e-5}], lr=lr, betas=(0.0, 0.99))
disc_optimizer = optim.Adam(disc_net.parameters(), lr=lr, betas=(0.0, 0.99))

# Lists to keep track of progress
G_losses = []
D_losses = []
iters = 0

# Monitor Progress
progress = list()
fixed_noise = torch.randn(100, z_dim, device=device)
fixed_labels = torch.Tensor([[i]*10 for i in range(10)]).view(100,).int().to(device)

# Training loop
gen_net.train()
disc_net.train()
step = int(math.log2(start_train_img_size / 4))
for num_epochs in progressive_epochs[step:]:
    alpha = 1e-7
    
    loader, dataset = get_loader(4*2**step)
    print("Curent image size: " + str(4*2**step))

    for epoch in range(num_epochs):
        # print(f"Epoch [{epoch + 1}/{num_epochs}")

        loop = tqdm(loader, leave=True)

        for batch_idx, (real_images, _) in enumerate(loop):
            real_images = real_images.to(device)
            current_batch_size = real_images.shape[0]
            noise = torch.randn(current_batch_size, z_dim).to(device)
            
            fake = gen_net(noise, alpha, step)
            real_output = disc_net(real_images, alpha, step)
            fake_output = disc_net(fake.detach(), alpha, step)

            grad_penalty = gradient_penalty(disc_net, real_images, fake, alpha, step, device)
            disc_loss = (-(torch.mean(real_output)-torch.mean(fake_output)) + lambda_gp*grad_penalty + (0.001)*torch.mean(real_output**2))

            disc_net.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            gen_fake = disc_net(fake, alpha, step)
            gen_loss = -torch.mean(gen_fake)

            gen_net.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            alpha += current_batch_size/(progressive_epochs[step]*0.5*len(dataset))
            alpha = min(alpha,1)

            loop.set_postfix(grad_penalty = grad_penalty.item(), disc_loss = disc_loss.item())
        
            # Training Update
            if batch_idx % 50 == 0:
                print(f"[{epoch}/{num_epochs}][{batch_idx}/{len(loader)}]\tLoss_D: {disc_loss.item()}\tLoss_G: {gen_loss.item()}")

            # Tracking loss
            G_losses.append(gen_loss.item())
            D_losses.append(disc_loss.item())

            # Tracking Generator Progress
            if (iters%10 == 0) or ((epoch == num_epochs-1) and (batch_idx == len(loader)-1)):
                with torch.no_grad():
                    fake = gen_net(fixed_noise, fixed_labels).detach().cpu()
                progress.append(torch_utils.make_grid(fake, padding=2, nrow=10, normalize=True))

            iters += 1

    generate_examples(gen_net, step)
    step += 1

# Save generator
torch.save(gen_net, gen_save_path)
torch.save(disc_net, disc_save_path)

# Plot Training Graph
fig1 = plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(training_plot_save_path)
plt.show()

# Progress Animation
fig2 = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in progress]
anim = animation.ArtistAnimation(fig2, ims, interval=1000, repeat_delay=1000, blit=True)
writervideo = animation.FFMpegWriter(fps=5)
anim.save(animation_save_path, writer=writervideo)
plt.close()