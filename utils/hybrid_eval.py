def aggregate_confidence(values, mode="min"):
    if not values:
        return 0.0

    if mode == "min":
        return min(values)

    if mode == "mean":
        return sum(values) / len(values)

    if mode == "product":
        result = 1.0
        for v in values:
            result *= v
        return result

    raise ValueError(f"Unknown aggregate mode '{mode}'")
