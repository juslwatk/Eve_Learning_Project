import torch
import torch.nn as nn
from Encryptor import Encryptor
class EveModel(nn.Module):
    def __init__(self, bit_length):
        super().__init__()
        self.bit_length = bit_length
        self.num_classes = 95
        self.model = nn.Sequential(
            nn.Linear(bit_length, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, bit_length * self.num_classes)
        )

    def forward(self, x):
        out = self.model(x)
        return out.view(-1, self.bit_length, self.num_classes)

    def save(self, path, epoch, optimizer, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, path)

    def load(self, path, optimizer):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'] + 1
    def test_on_random_messages(self, sentences, bit_length, device, num_tests=25):
        import random
        self.eval()
        correct_sentences = 0
        total_chars = 0
        correct_chars = 0

        print(f"\nTesting Eve on {num_tests} messages...\n")

        with torch.no_grad():
            for i in range(num_tests):
                original_text = random.choice(sentences)
                original_bytes = original_text.encode("utf-8")
                if len(original_bytes) < bit_length:
                    original_bytes += b' ' * (bit_length - len(original_bytes))

                cipher = Encryptor.easy_shift_encrypt(original_bytes)
                input_tensor = Encryptor.bytes_to_tensor(cipher).unsqueeze(0).to(device)
                output_tensor = self(input_tensor)
                recovered_bytes = Encryptor.tensor_to_bytes_classification(output_tensor)
                recovered_text = Encryptor.safe_decode(recovered_bytes).strip()

                match = recovered_bytes == original_bytes

                print(f"Test #{i+1}")
                print(f"Original : {original_text}")
                print(f"Recovered: {recovered_text}")
                print(f"Match    : {"Success" if match else "Fail"}\n")

                if match:
                    correct_sentences += 1

                correct_chars += sum(a == b for a, b in zip(original_bytes, recovered_bytes))
                total_chars += len(original_bytes)

        sentence_acc = (correct_sentences / num_tests) * 100
        char_acc = (correct_chars / total_chars) * 100

        print(f"Sentence Accuracy: {sentence_acc:.2f}% ({correct_sentences}/{num_tests})")
        print(f"Character Accuracy: {char_acc:.2f}%")

        self.train()  # Return to training mode if needed
