import os


def get_filename(crypto, classifier, num_classes, seed):
    filename = (
        f"{crypto}_labels_{classifier}_{num_classes}_{seed}.csv"
        if seed is not None
        else f"{crypto}_labels_{classifier}_{num_classes}.csv"
    )
    return filename


def get_trading_stats_filename(classifier, num_classes, seed):
    filename = (
        f"stats_{classifier}_{num_classes}_{seed}.csv"
        if seed is not None
        else f"stats_{classifier}_{num_classes}.csv"
    )
    return filename


def get_equity_filename(classifier, num_classes, seed):
    filename = (
        f"{classifier}_{num_classes}_equity_{seed}.csv"
        if seed is not None
        else f"{classifier}_{num_classes}_equity.csv"
    )
    return filename


def get_results_filename(classifier, num_classes, seed):
    filename = (
        f"results_{classifier}_{num_classes}.csv"
        if seed is None
        else f"results_{classifier}_{num_classes}_{seed}.csv"
    )
    return filename


def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
