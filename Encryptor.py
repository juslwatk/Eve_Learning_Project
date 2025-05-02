import torch
import random

class Encryptor:
    @staticmethod
    def easy_shift_encrypt(msg: bytes, max_shift=3):
        shift = random.randint(1, max_shift)
        return bytes([(m + shift - 32) % 95 + 32 for m in msg])

    @staticmethod
    def bytes_to_tensor(b: bytes):
        return torch.tensor([(min(126, max(32, x)) - 32) / 94 for x in b], dtype=torch.float32)

    @staticmethod
    def bytes_to_class_tensor(b: bytes):
        return torch.tensor([min(126, max(32, x)) - 32 for x in b], dtype=torch.long)

    @staticmethod
    def tensor_to_bytes_classification(t: torch.Tensor):
        preds = torch.argmax(t, dim=2).squeeze(0).detach().cpu().numpy()
        return bytes([x + 32 for x in preds])

    @staticmethod
    def safe_decode(b: bytes):
        try:
            return b.decode('utf-8')
        except UnicodeDecodeError:
            return "(Unreadable Output)"
