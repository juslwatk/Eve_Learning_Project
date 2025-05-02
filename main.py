
import torch
import os
from EveModel import EveModel  
from Trainer import Trainer  
from Encryptor import Encryptor 
import random

# Load data
def read_sentences_from_file(filename):
    sentences = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            sentences = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    return sentences

# Configuration
FILENAME = "android_dev.txt"
CHECKPOINT_PATH = "eve_checkpoint.pth"
TOTAL_EPOCHS = 800
BATCH_SIZE = 64
LEARNING_RATE = 0.001



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentences = read_sentences_from_file(FILENAME)
    bit_length = max(len(s.encode("utf-8")) for s in sentences)

    model = EveModel(bit_length)
    trainer = Trainer(
        model=model,
        sentences=sentences,
        bit_length=bit_length,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
    )

    trainer.resume()
    trainer.train(TOTAL_EPOCHS)

    # Optional: quick evaluation
    model.eval()
    test_text = random.choice(sentences)
    test_bytes = test_text.encode("utf-8")
    if len(test_bytes) < bit_length:
        test_bytes += b' ' * (bit_length - len(test_bytes))

    cipher = Encryptor.easy_shift_encrypt(test_bytes)
    input_tensor = Encryptor.bytes_to_tensor(cipher).unsqueeze(0).to(device)
    output = model(input_tensor)
    recovered = Encryptor.tensor_to_bytes_classification(output)

    print("\nTest Example")
    print("Original:", test_text)
    print("Recovered:", Encryptor.safe_decode(recovered))
    print("Match:", recovered == test_bytes)
    # model.test_on_random_messages(sentences, bit_length, device, num_tests=25)


if __name__ == "__main__":
    main()
