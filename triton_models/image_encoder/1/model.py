import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            images = pb_utils.get_input_tensor_by_name(request, "images").as_numpy()
            if images.ndim != 4 or images.shape[1] != 3:
                raise pb_utils.TritonModelException(
                    "images must have shape [batch, 3, height, width]"
                )

            height_midpoint = images.shape[2] // 2
            width_midpoint = images.shape[3] // 2
            channel_mean = images.mean(axis=(2, 3))
            channel_std = images.std(axis=(2, 3))
            quadrants = [
                images[:, :, :height_midpoint, :width_midpoint],
                images[:, :, :height_midpoint, width_midpoint:],
                images[:, :, height_midpoint:, :width_midpoint],
                images[:, :, height_midpoint:, width_midpoint:],
            ]
            quadrant_means = [quadrant.mean(axis=(2, 3)) for quadrant in quadrants]
            embeddings = np.concatenate(
                [channel_mean, channel_std, *quadrant_means],
                axis=1,
            ).astype(np.float32)
            output = pb_utils.Tensor("embeddings", embeddings)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
        return responses
