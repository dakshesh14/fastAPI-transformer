from transformers import (
    AutoModel,
    T5Tokenizer,
    AutoTokenizer,
    T5ForConditionalGeneration,
)
import torch
from fastapi import FastAPI, APIRouter

# cors headers
from fastapi.middleware.cors import CORSMiddleware


# types
from typing import List
from pydantic import BaseModel


class Content(BaseModel):
    idx: int
    content: str


class Similarity(BaseModel):
    main_content: Content
    other_contents: List[Content]


class Summary(BaseModel):
    content: str
    max_length: int = 250


model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=1024)

sim_model = AutoModel.from_pretrained("bert-base-uncased")
sim_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# chunk size will be used to break large content
chunk_size = 512


app = FastAPI()
router = APIRouter()

origin = [
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@router.post("/summarize")
def summarize(data: Summary):

    text = data.content
    max_length = data.max_length

    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt",
        max_length=2440,
        truncation=True
    )
    outputs = model.generate(
        inputs, max_length=max_length, min_length=40,
        length_penalty=2.0, num_beams=4, early_stopping=True
    )

    response = {
        "summary": tokenizer.decode(outputs[0], skip_special_tokens=True)
    }

    return response


@router.post("/similarity")
def similarity(data: Similarity):

    main_content = data.main_content.content
    other_contents = data.other_contents

    chunks = [
        main_content[i:i+chunk_size]
        for i in range(0, len(main_content), chunk_size)
    ]

    # Get the embedding of each chunk of the main content
    chunk_embeddings = []
    for chunk in chunks:
        # Tokenize and convert to PyTorch tensor
        input_ids = torch.tensor(sim_tokenizer.encode(
            chunk, add_special_tokens=True
        )).unsqueeze(0)
        # Get the embedding
        with torch.no_grad():
            last_hidden_states = sim_model(input_ids)[0]
            chunk_embedding = torch.mean(last_hidden_states, dim=1).squeeze()
        chunk_embeddings.append(chunk_embedding)

    # looping thru list of other contents, tokenizing them and getting embedding for each chunks
    content_embeddings = []
    for content in other_contents:

        content = content.content

        # Split the content into smaller chunks
        content_chunks = [
            content[i:i+chunk_size]for i in range(0, len(content), chunk_size)
        ]
        # Get the embedding of each chunk
        chunk_embeddings = []
        for chunk in content_chunks:
            # Tokenize and convert to PyTorch tensor
            input_ids = torch.tensor(sim_tokenizer.encode(
                chunk, add_special_tokens=True)
            ).unsqueeze(0)
            # Get the embedding
            with torch.no_grad():
                last_hidden_states = sim_model(input_ids)[0]
                chunk_embedding = torch.mean(
                    last_hidden_states, dim=1
                ).squeeze()
            chunk_embeddings.append(chunk_embedding)
        # Combine the embeddings of all chunks into a single embedding for the content
        content_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
        content_embeddings.append(content_embedding)

    # Calculate the cosine similarity between the main content and each content in the list
    similarities = []
    for content_embedding in content_embeddings:
        chunk_similarities = []
        for chunk_embedding in chunk_embeddings:
            # Add a new dimension to chunk_embedding to match the shape of content_embedding
            chunk_embedding_with_batch = chunk_embedding.unsqueeze(0)
            chunk_similarity = torch.nn.functional.cosine_similarity(
                chunk_embedding_with_batch, content_embedding, dim=0
            )
            chunk_similarities.append(chunk_similarity)
        similarity = torch.mean(torch.stack(chunk_similarities))
        similarities.append(similarity)

    sorted_contents = [
        x.idx for _, x in sorted(zip(similarities, other_contents))
    ]

    return sorted_contents


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    # run uvicorn with reload
    uvicorn.run("main:app", port=8127, reload=True)
