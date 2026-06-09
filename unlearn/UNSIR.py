
import torch
import torch.nn.functional as F
import numpy as np
class UNSIR_noise(torch.nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad = True)
        
    def forward(self):
        return self.noise
    
def UNSIR_noise_train(noise, model, forget_class_label, num_epochs, noise_batch_size, device='cuda'):
    opt = torch.optim.Adam(noise.parameters(), lr = 0.1)
  
    for epoch in range(num_epochs):
        total_loss = []
        inputs = noise()
        labels = torch.zeros(noise_batch_size).to(device)+forget_class_label
        outputs = model(inputs)
        loss = -F.cross_entropy(outputs, labels.long()) + 0.1*torch.mean(torch.sum(inputs**2, [1, 2, 3]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss.append(loss.cpu().detach().numpy())
        if epoch%5 == 0:
            print("Loss: {}".format(np.mean(total_loss)))
            
    return noise

def UNSIR_create_noisy_loader(noise, forget_class_label, retain_samples, batch_size, num_noise_batches=80, device='cuda'):
    
    noisy_data = []
    for i in range(num_noise_batches):
        batch = noise()
        for i in range(batch[0].size(0)):
            noisy_data.append((batch[i].detach().cpu(), torch.tensor(forget_class_label), \
                               torch.tensor(forget_class_label)))

    other_samples = []
    for i in range(len(retain_samples)):
        other_samples.append((retain_samples[i][0].cpu(), torch.tensor(retain_samples[i][2]),\
                            torch.tensor(retain_samples[i][2])))
    noisy_data += other_samples
    noisy_loader = torch.utils.data.DataLoader(noisy_data, batch_size=batch_size, shuffle = True)
    
    return noisy_loader