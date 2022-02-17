import base64
import requests
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

        auto = request.form.get("auto")
        if auto is None:
            auto = True
        else:
            auto = bool(int(auto))

        num_generations = request.form.get("numGenerations")
        if num_generations is None:
            num_generations = 1
        else:
            num_generations = int(num_generations)

        cond_img = request.form.get("condImg")
        if cond_img is not None:
            response = requests.get(cond_img)
            cond_img = Image.open(BytesIO(response.content)).convert("RGB")

        if auto:
            param_dict = None
        else:
            resolution = request.form.get("resolution").split(",")
            if resolution is None:
                resolution = (400, 400)
            else:
                resolution = [int(res) for res in resolution]

            strength = request.form.get("strength")
            if strength is None:
                strength = 0.3
            else:
                strength = float(strength)

            num_iterations = request.form.get("numIterations")
            if num_iterations is None:
                num_iterations = 30
            else:
                num_iterations = int(num_iterations)

            do_upscale = request.form.get("numIterations")
            if do_upscale is None:
                do_upscale = False
            else:
                do_upscale = bool(int(do_upscale))

            num_crops = request.form.get("realism")
            if num_crops is None:
                num_crops = 64
            else:
                num_crops = int(num_crops)

            param_dict = {
                "resolution": resolution,
                "lr": strength,
                "num_iterations": num_iterations,
                "do_upscale": do_upscale,
                "num_crops": num_crops,
            }

        print("param dict", param_dict)

        nft_img_list = cosmic_nft.generate_nfts_from_prompt(
            prompt_list=prompt_list,
            num_nfts=num_generations,
            cond_img=cond_img,
            auto=auto,
            param_dict=param_dict,
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
    port=8008,
)
