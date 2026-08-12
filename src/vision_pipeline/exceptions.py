class VisionPipelineError(Exception):
    pass


class ConfigurationError(VisionPipelineError):
    pass


class FactoryError(VisionPipelineError):
    pass


class ModelArtifactError(VisionPipelineError):
    pass


class InferenceError(VisionPipelineError):
    pass


class StreamError(VisionPipelineError):
    pass
