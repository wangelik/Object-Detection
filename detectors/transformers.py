# import table
import os
from typing import Optional

from PIL import Image, ImageFont
from transformers import DetrForObjectDetection, DetrImageProcessor, pipeline

from .utils.visualize import add_bounding_box, v_print


def detect(
    images: list,
    object: Optional[str],
    score: bool = True,
    fontsize: int = 12,
    output_dir: str = "output/transformers",
    verbose: bool = True,
):
    """Object detector using detection transformer technology

    Arguments:
        images (list): list of image paths to analyze
        object (str): optional object type to exclusively detect
        score (bool): to show score for detection or not
        fontsize (int): fontsize for text information
        output_dir (str): directory for image outputs
        verbose (bool): print state information or not

    Returns:
        detections (list): list of detections per image

    """

    # settings
    font = ImageFont.truetype("tahoma.ttf", fontsize)
    detections = []

    # loading detectors and creating pipeline
    feature_extractor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
    object_detector = pipeline(
        "object-detection", model=model, feature_extractor=feature_extractor
    )

    # load all images
    for img in images:

        # extract image name
        img_name = os.path.basename(img).split(".")[0]

        # load image data
        with Image.open(img) as im:

            # convert image to RGBA
            im = im.convert("RGBA")
            v_print(f"[+] Loaded image {img_name}", verbose)

            # perform detection and store results
            objects = object_detector(im)
            v_print(f"[-] Found {len(objects)} detections")
            detections.append(objects)

            # analyze all objects
            for obj in objects:

                # check for labels and scores
                acc = obj["score"] if score else None
                label = obj["label"] if not object else None
                box = obj["box"]

                # if specific object is provided: filtering
                if object and obj["label"] != object:
                    continue

                # add bounding box and text information
                im = add_bounding_box(
                    im,
                    [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
                    label,
                    acc,
                    font,
                )

            # save new image
            os.makedirs(output_dir, exist_ok=True)
            im.save(f"{output_dir}/{img_name}.png")

    # return all detections
    return detections
