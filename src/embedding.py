from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from typing import List, Literal


class JinaEmbeddings:
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v3",
        device: str | None = None,
        vector_size: int = 1024,
    ):
        if not device:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = device
        print(f"Using device: {device}")

        self.vector_size = vector_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,  # local_files_only=True
        )
        print(f"Jina Tokenizer loaded.")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            device
        )
        print(f"Jina Model loaded.")

    def encode(
        self,
        sentences: List[str] | str,
        task: Literal[
            "retrieval.query",
            "retrieval.passage",
            "separation",
            "classification",
            "text-matching",
        ],
    ):
        if type(sentences) != list:
            sentences = [sentences]
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        task_id = self.model._adaptation_map[task]
        adapter_mask = torch.full((len(sentences),), task_id, dtype=torch.int32)

        with torch.inference_mode():
            model_output = self.model(**encoded_input, adapter_mask=adapter_mask)

        embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        embeddings = embeddings.cpu().numpy()
        self._empty_cache()

        return embeddings

    def _empty_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


if __name__ == "__main__":
    jina = JinaEmbeddings(cache_dir="./models")
