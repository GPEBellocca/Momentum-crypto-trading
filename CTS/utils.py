def get_filename(crypto, classifier, num_classes, seed):
    filename = (
        f"{crypto}_labels_{classifier}_{num_classes}_{seed}.csv"
        if seed is not None
        else f"{crypto}_labels_{classifier}_{num_classes}.csv"
    )
    return filename