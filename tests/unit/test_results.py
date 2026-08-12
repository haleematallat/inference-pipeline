from vision_pipeline.common.results import InferResult, Prediction


def test_prediction_round_trip():
    prediction = Prediction("blue", 0.9, accepted=True, rank=1)
    restored = Prediction.from_dict(prediction.to_dict())
    assert restored.class_name == "blue"
    assert restored.similarity == 0.9
    assert restored.accepted is True


def test_infer_result_round_trip():
    result = InferResult(
        timestamp=12,
        source="image.ppm",
        predictions=[Prediction("amber", 0.8, True)],
        metadata={"runner": "few-shot"},
    )
    restored = InferResult.from_dict(result.to_dict())
    assert restored.timestamp == 12
    assert restored.predictions[0].class_name == "amber"
    assert restored.metadata["runner"] == "few-shot"
