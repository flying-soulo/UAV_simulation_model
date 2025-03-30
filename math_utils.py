def wrap(value, min_val, max_val):
    """
    Clamps a given value within the specified range [min_val, max_val].

    This function ensures that the input value does not exceed the maximum 
    or fall below the minimum of the specified range.

    Args:
        value (float): The value to be clamped.
        min_val (float): The minimum allowable value.
        max_val (float): The maximum allowable value.

    Returns:
        float: The clamped value within the range [min_val, max_val].
    """
    return max(min_val, min(max_val, value))