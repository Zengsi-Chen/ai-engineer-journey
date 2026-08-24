from dataclasses import dataclass


@dataclass
class EpochResult:

    epoch: int

    train_loss: float

    val_loss: float

    learning_rate: float

    is_best: bool = False

    should_stop: bool = False


@dataclass
class TrainingHistory:

    train_loss: list
    val_loss: list
    learning_rate: list

    def to_dict(self):

        return {

            "train_loss":
                self.train_loss,

            "val_loss":
                self.val_loss,

            "learning_rate":
                self.learning_rate
        }


    def add(
        self,
        epoch_result: EpochResult
    ):

        self.train_loss.append(
            epoch_result.train_loss
        )

        self.val_loss.append(
            epoch_result.val_loss
        )

        self.learning_rate.append(
            epoch_result.learning_rate
        )

    @property
    def epochs(self):

        return len(
            self.train_loss
        )