from scenario_analysis_common import run_scenario_analysis


def main():
    run_scenario_analysis(7, script_dir=__import__("os").path.dirname(__file__))


if __name__ == "__main__":
    main()
