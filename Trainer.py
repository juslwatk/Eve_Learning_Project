import torch
import torch.nn as nn
import torch.optim as optim
import random
from Encryptor import Encryptor
import os
class Trainer:
    def __init__(self, model, sentences, bit_length, checkpoint_path, device, batch_size=64, lr=0.001):
        self.model = model.to(device)
        self.sentences = sentences
        self.bit_length = bit_length
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.start_epoch = 0

    def resume(self):
        if os.path.exists(self.checkpoint_path):
            self.start_epoch = self.model.load(self.checkpoint_path, self.optimizer)
            print(f"Resuming from epoch {self.start_epoch}")
        else:
            print("No checkpoint found. Starting fresh training.")

    def train(self, epochs):
        self.model.train()
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            total_loss = 0.0
            for _ in range(self.batch_size):
                batch_x, batch_y = [], []
                for _ in range(self.batch_size):
                    msg_text = random.choice(self.sentences)
                    msg_bytes = msg_text.encode('utf-8')
                    if len(msg_bytes) < self.bit_length:
                        msg_bytes += b' ' * (self.bit_length - len(msg_bytes))

                    cipher = Encryptor.easy_shift_encrypt(msg_bytes)
                    x = Encryptor.bytes_to_tensor(cipher)
                    y = Encryptor.bytes_to_class_tensor(msg_bytes)
                    batch_x.append(x)
                    batch_y.append(y)

                x_tensor = torch.stack(batch_x).to(self.device)
                y_tensor = torch.stack(batch_y).to(self.device)

                output = self.model(x_tensor)
                loss = self.loss_fn(output.view(-1, 95), y_tensor.view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / self.batch_size
            print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

            if (epoch + 1) % 10 == 0:
                self.model.save(self.checkpoint_path, epoch, self.optimizer, loss)
                print(f"Checkpoint saved at epoch {epoch+1}")
