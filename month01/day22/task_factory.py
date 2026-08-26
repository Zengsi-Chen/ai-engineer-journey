from tasks import (
    BinaryClassificationTask,
    MulticlassClassificationTask,
    RegressionTask,
)


def create_task(config):

    if config.task_type == "binary_classification":

        return BinaryClassificationTask(
            threshold=config.threshold
        )

    if config.task_type == "multiclass_classification":

        return MulticlassClassificationTask()

    if config.task_type == "regression":

        return RegressionTask()

    raise ValueError(
        f"Unknown task: {config.task_type}"
    )