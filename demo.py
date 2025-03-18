import argparse
from mesh_estimator import HumanMeshEstimator


def get_model_disk_size(model, unit="MB"):
    """
    Calculates the disk size of a PyTorch model (parameters + buffers).

    Args:
        model (torch.nn.Module): The PyTorch model.
        unit (str, optional): The unit for disk size. Can be 'B', 'KB', 'MB', or 'GB'.
                              Defaults to 'MB'.

    Returns:
        float: The model size on disk in the specified unit.
    """

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_bytes = param_size + buffer_size

    if unit == "KB":
        disk_size = total_size_bytes / 1024
    elif unit == "MB":
        disk_size = total_size_bytes / (1024 * 1024)
    elif unit == "GB":
        disk_size = total_size_bytes / (1024 * 1024 * 1024)
    else:
        disk_size = total_size_bytes

    return disk_size


def make_parser():
    parser = argparse.ArgumentParser(description="CameraHMR Regressor")
    parser.add_argument(
        "--image_folder",
        "--image_folder",
        type=str,
        help="Path to input image folder.",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        "--output_folder",
        type=str,
        help="Path to folder output folder.",
        required=True,
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    estimator = HumanMeshEstimator()
    estimator.run_on_images(args.image_folder, args.output_folder)

    print("Estimator:")
    print("size: ", get_model_disk_size(estimator.model, unit="MB"), "MB")
    print(estimator.model)
    print("-" * 20)
    print()
    print("Detector:")
    print("size: ", get_model_disk_size(estimator.detector.model, unit="MB"), "MB")
    print(estimator.detector.model)
    print("-" * 20)
    print()
    print("Camera Model:")
    print("size: ", get_model_disk_size(estimator.cam_model, unit="MB"), "MB")
    print(estimator.cam_model)


if __name__ == "__main__":
    main()
