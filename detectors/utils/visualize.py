# import table
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


def add_bounding_box(
    image: Image,
    dims: list[int],
    label: Optional[str],
    score: Optional[float],
    font: Optional[ImageFont],
    box_color: tuple = (0, 172, 255),
    box_alpha: float = 0.25,
    text_color: tuple = (0, 0, 0),
    text_bg_color: tuple = (255, 255, 255),
):
    """Drawing bounding box and text information on image objects

    Arguments:
        image (Image): PIL image object
        dims (list): list of bounding box coordinates (x1, y1, x2, y2)
        label (str): optional label to show next to box
        score (float): optional accuracy to show next to label
        font (ImageFont): optional ImageFont font object
        box_color (tuple): RGB colors of the bounding box
        box_alpha (float): transpareny of the bounding box fill
        text_color (tuple): RGB colors of the text information
        text_bg_color (tuple): RGB colors of the text background

    Returns:
        image (Image): enriched PIL image object

    """

    # create new RGBA image layer
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # create text information
    text = ""
    if label:
        text += label
    if score:
        if text:
            text += f" | {score:.2f}"
        else:
            text += f"{score:.2f}"

    # draw bounding box
    draw.rounded_rectangle(
        (dims[0], dims[1], dims[2], dims[3]),
        fill=box_color + (int(box_alpha * 255),),
        outline=box_color + (255,),
        width=2,
        radius=5,
    )

    # insert text information
    if text:
        left, top, right, bottom = draw.textbbox(
            (dims[0], dims[3] + 4), text, font=font
        )
        draw.rounded_rectangle(
            (left, top - 4, right + 7, bottom + 3), fill=text_bg_color, radius=2
        )
        draw.text((dims[0] + 4, dims[3] + 4), text, font=font, fill=text_color)

    # compose and return result
    image = Image.alpha_composite(image, overlay)
    return image


def v_print(input: str, verbose: bool = True):
    """Verbosity dependent stdout utility"""

    if verbose:
        print(input)
