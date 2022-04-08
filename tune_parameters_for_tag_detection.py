from email.policy import default
from mimetypes import suffix_map
import cv2
import click
from pathlib import Path
from pupil_apriltags import Detector
from scripts.image import bgr2gray
from scripts.april import extract_corner_and_id_list_apriltags
from scripts.utils import load_toml
from scripts.intrinsic import Intrinsic
import optuna
from tqdm import tqdm
from functools import reduce


SCRIPT_DIR = str(Path().parent.resolve())


def generate_tag_detector(quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=0.25) -> Detector:
    return Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=1,
        decode_sharpening=decode_sharpening,
    )


def total_number_of_detected_markers(at_detector, image_path, intrinsic_parameter, tag_size):
    # total_number_of_tags = 0
    detected_tag_id_list = []
    for image_path in image_pathes:
        image = cv2.imread(image_path)
        tags = at_detector.detect(
            bgr2gray(image), estimate_tag_pose=False, camera_params=intrinsic_parameter.parameters, tag_size=tag_size
        )
        _, april_ids = extract_corner_and_id_list_apriltags(tags)
        detected_tag_id_list.append(april_ids)
    total_number_of_tags = sum([len(tag_ids) for tag_ids in detected_tag_id_list])
    return total_number_of_tags


# @TODO Reduce global variable
image_pathes = []
tag_size = None
intrinsic_parameter = Intrinsic()

@click.command()
@click.option("--input-image-dir", "-i", type=str, default=f"{SCRIPT_DIR}/data")
@click.option("--parameter-path", "-c", type=str, default=f"{SCRIPT_DIR}/config/parameters.toml")
def main(input_image_dir, parameter_path):
    global image_pathes
    global intrinsic_parameter
    global tag_size

    """Get image pathes"""
    image_pathes = [str(path) for path in Path(input_image_dir).glob("*") if path.suffix in [".jpg", ".png"]]
    print(f"number of taget images: {len(image_pathes)}")

    """ Load camera parameters and detection parameters
    """
    config_toml = load_toml(parameter_path)
    intrinsic_parameter.set(config_toml)
    tag_size = config_toml["apriltags"]["tag_size"]

    """ Optuna tuning
    """

    # @TODO Reduce global variable
    def objective(trial) -> int:
        global image_pathes
        global intrinsic_parameter
        global tag_size

        quad_decimate = trial.suggest_uniform("quad_decimate", 1.0, 4.0)
        quad_sigma = trial.suggest_uniform("quad_sigma", 0.0, 15.0)
        decode_sharpening = trial.suggest_uniform("decode_sharpening", 0.10, 0.5)
        at_detector = generate_tag_detector(quad_decimate, quad_sigma, decode_sharpening)
        num_tag = total_number_of_detected_markers(at_detector, image_pathes, intrinsic_parameter, tag_size)
        return num_tag

    # sampler = optuna.samplers.CmaEsSampler()
    sampler = optuna.samplers.TPESampler()
    # sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=100)
    print(f"params:{study.best_params}")
    print(f"value:{study.best_value}")


if __name__ == "__main__":
    main()
