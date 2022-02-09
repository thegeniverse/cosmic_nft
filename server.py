import base64
from io import BytesIO

from flask import Flask, request
from PIL import Image

from cosmic import CosmicNFT

app = Flask(__name__)


def base64_to_PIL(base64_encoding: str):
    return Image.open(BytesIO(base64.b64decode(base64_encoding)))


def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    data_uri = base64.b64encode(buffer.read()).decode('ascii')

    return data_uri


cosmic_nft = CosmicNFT()


@app.route(
    '/generate',
    methods=['POST'],
)
def generate():
    try:
        prompt_list = request.form.get("text").split("-")
        num_generations = request.form.get("numGenerations")
        cond_img = request.form.get("condImg")

        if num_generations is None:
            num_generations = 1

        print("prompt list", prompt_list)
        print("num generations", num_generations)
        print("len cond img", len(cond_img))

        if cond_img is not None:
            cond_img = base64_to_PIL(cond_img, )

        nft_img_list = cosmic_nft.generate_nfts_from_prompt(
            prompt_list,
            num_generations,
            cond_img,
        )

        result_dict = {
            "success": True,
            "imgArray": [pil_to_base64(img) for img in nft_img_list],
        }

    except Exception as e:
        result_dict = {
            "success": False,
            "error": repr(e),
        }

    return result_dict


app.run(
    host="0.0.0.0",
    port=8000,
)
