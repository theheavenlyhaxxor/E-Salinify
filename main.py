from typing import Union
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/HAND_SIGNS", StaticFiles(directory="HAND_SIGNS"), name="HAND_SIGNS")

@app.get('/translate/{word}/')
async def convert(word: str):
    letters = [char.lower() for char in word if char.isalpha()]

    image_paths = [f"/HAND_SIGNS/{letter}.png" for letter in letters]
    return {"converted_words" : image_paths}