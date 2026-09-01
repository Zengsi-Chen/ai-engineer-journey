from domain_experiment import (
    create_domain_shift_experiments,
)

def test_creates_three_experiments():

    experiments = (
        create_domain_shift_experiments()
    )

    assert len(experiments) == 3

def test_experiment_names():

    experiments = (
        create_domain_shift_experiments()
    )

    names = [
        experiment.name
        for experiment in experiments
    ]

    assert names == [
        "no_shift",
        "moderate_shift",
        "severe_shift",
    ]

def test_no_shift_has_no_transformation():

    experiment = (
        create_domain_shift_experiments()[0]
    )

    config = experiment.shift_config

    assert config.brightness == 0.0
    assert config.contrast == 0.0
    assert config.grayscale is False
    assert config.noise_std == 0.0

def test_shift_strength_increases():

    (
        no_shift,
        moderate_shift,
        severe_shift,
    ) = create_domain_shift_experiments()

    assert (
        moderate_shift.shift_config.brightness
        >
        no_shift.shift_config.brightness
    )

    assert (
        severe_shift.shift_config.brightness
        >
        moderate_shift.shift_config.brightness
    )

    assert severe_shift.shift_config.grayscale

