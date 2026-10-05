"""Where training resumes from a checkpoint: next epoch after an end-of-epoch save,
the same epoch (skipping the batches already trained) after a mid-epoch save."""

from ppocr.utils.save_load import resume_position


def test_end_of_epoch_checkpoint_resumes_at_next_epoch():
    assert resume_position({"epoch": 4, "global_step": 900}) == (5, 0)


def test_mid_epoch_checkpoint_resumes_same_epoch_and_skips_trained_batches():
    assert resume_position({"epoch": 4, "global_step": 950, "step_in_epoch": 50}) == (4, 50)


def test_zero_steps_in_epoch_is_end_of_epoch():
    assert resume_position({"epoch": 4, "step_in_epoch": 0}) == (5, 0)


def test_no_epoch_means_start_from_scratch():
    assert resume_position({}) == (None, 0)
