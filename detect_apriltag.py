from mimetypes import suffix_map
import cv2
import click
from pathlib import Path
from pupil_apriltags import Detector
from scripts.image import bgr2gray
from scripts.april import extract_corner_and_id_list_apriltags
from scripts.utils import load_toml
from scripts.intrinsic import Intrinsic
from tqdm import tqdm


SCRIPT_DIR = str(Path().parent.resolve())


def generate_tag_detector(quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25) -> Detector:
    return Detector(
        families="tag36h11", nthreads=2, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25
    )


@click.command()
@click.option("--input-image-dir", "-i", type=str, default=f"{SCRIPT_DIR}/data")
@click.option("--parameter-path", "-c", type=str, default=f"{SCRIPT_DIR}/config/parameters.toml")
def main(input_image_dir, parameter_path):
    """Get image pathes"""
    image_pathes = [str(path) for path in Path(input_image_dir).glob("*") if path.suffix in [".jpg", ".png"]]
    print(f"number of taget images: {len(image_pathes)}")

    """ Load camera parameters and detection parameters
    """
    config_toml = load_toml(parameter_path)
    intrinsic_parameter = Intrinsic()
    intrinsic_parameter.set(config_toml)
    tag_size = config_toml["apriltags"]["tag_size"]

    """ Generate a tag detector module
    """
    at_detector = generate_tag_detector(quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)

    """ Tag detection
    """
    for image_path in tqdm(image_pathes):
        image = cv2.imread(image_path)
        tags = at_detector.detect(
            bgr2gray(image), estimate_tag_pose=True, camera_params=intrinsic_parameter.parameters, tag_size=tag_size
        )
        april_corners, april_ids = extract_corner_and_id_list_apriltags(tags)
        print(f"number of detected markers: {len(april_ids)}")


if __name__ == "__main__":
    main()
