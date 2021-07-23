import logging

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import click
import spacy
import contextualSpellCheck


logger = logging.getLogger("make_request")

nlp = spacy.load("en_core_web_sm")
contextualSpellCheck.add_to_pipe(nlp)


def convert_pdf_to_image(document_path, dpi=350):
    images = []
    images.extend(
        list(
            map(
                lambda image: cv2.cvtColor(np.asarray(image), code=cv2.COLOR_BGR2RGB),
                convert_from_path(document_path, dpi=dpi),
            )
        )
    )
    return images


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def postprocess_text(text):
    text = text[text.conf != -1]
    answer = []
    lines = text.groupby("block_num")["text"].apply(lambda x: " ".join(list(x)))
    conf = text.groupby(["block_num"])["conf"].mean()
    logger.info("Postprocessing text")
    for (_, conf_row), (_, line_row) in zip(
        conf.to_frame().iterrows(), lines.to_frame().iterrows()
    ):
        if conf_row["conf"] < 90.0:
            doc = nlp(line_row["text"])
            if doc._.performed_spellCheck:
                line = doc._.outcome_spellCheck
                answer.append(line)
            else:
                answer.append(line_row["text"])
        else:
            answer.append(line_row["text"])
    return "\n".join(answer)


@click.command()
@click.option(
    "--input",
    default="./data/raw/test_scan.jpg",
    help="Input file path. Must be PNG/JPEG/PDF",
)
@click.option(
    "--output", default="./data/processed/test_scan.text", help="Output text file path"
)
@click.option("--verbose", is_flag=True, help="Output detailed logs")
def predict(input, output, verbose):
    if verbose:
        logging.basicConfig(
            format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO
        )
    logger.info("Program stars")
    if input.endswith(".pdf"):
        logger.info("Input file has pdf extension")
        logger.info("Converting pdf to images")
        images = convert_pdf_to_image(input)
        answer = []
        for image in images:
            logger.info("Processing image")
            image_processed = get_grayscale(image)
            image_processed = thresholding(image_processed)
            text = pytesseract.image_to_data(image_processed, output_type="data.frame")
            logger.info("Postprocessing text")
            text = postprocess_text(text)
            answer.append(text)
        logger.info("Writing extracted text to the file")
        with open(output, "w") as fout:
            fout.write("\n\n".join(answer))
    elif input.endswith(".jpg") or input.endswith(".png"):
        logger.info("Input file has jpg or png extension")
        image = cv2.imread(input)
        image = cv2.cvtColor(np.asarray(image), code=cv2.COLOR_BGR2RGB)
        logger.info("Processing images")
        image_processed = get_grayscale(image)
        image_processed = thresholding(image_processed)
        text = pytesseract.image_to_data(image_processed, output_type="data.frame")
        answer = postprocess_text(text)
        logger.info("Writing extracted text to the file")
        with open(output, "w") as fout:
            fout.write(answer)
    else:
        logger.warning(
            "The input file must be in pdf/jpg/png format and the file name must end in the corresponding format!"
        )


if __name__ == "__main__":
    predict()
